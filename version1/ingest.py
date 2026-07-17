"""
ingest.py — Indexing Pipeline (run this ONCE before using the assistant)
==========================================================================

WHAT THIS FILE DOES
--------------------
This script builds the searchable "knowledge base" that Version 1's RAG
pipeline retrieves from. It:

1. Reads every .txt file in the `emails/` folder.
2. Splits each email into overlapping chunks (see WHY CHUNKING below).
3. Converts each chunk into a numeric vector ("embedding") using a
   Sentence-Transformers model.
4. Stores those vectors in a FAISS index (a fast similarity-search
   structure) and saves it to disk, along with a metadata file that maps
   each vector back to its original text and source file.

WHY INDEXING HAPPENS ONLY ONCE
--------------------------------
Generating embeddings is comparatively expensive (loading a model,
running a forward pass per chunk) and the source emails rarely change.
So we do this work once, up front, and persist the result:

    emails/*.txt  --[ingest.py, run once]-->  vector_store/index.faiss
                                                vector_store/metadata.json

At *query time* (in rag.py), we only need to embed the user's single
query — a few milliseconds — and then search the already-built index.
This separation (slow "build" phase vs. fast "query" phase) is the
standard pattern behind every real-world RAG system.

If you add, remove, or edit files in `emails/`, re-run this script to
rebuild the index:

    python ingest.py
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, TypedDict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import SETTINGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("email_assistant.ingest")


class Chunk(TypedDict):
    """A single unit of text that will be embedded and indexed."""
    text: str
    source: str
    chunk_id: int


def load_emails(email_folder: str) -> dict[str, str]:
    """
    Read every .txt file in the given folder.

    Parameters:
        email_folder (str): path to the folder containing sample emails.

    Returns:
        dict[str, str]: mapping of {filename: raw_text_content}.

    Raises:
        FileNotFoundError: if the folder does not exist.

    Example:
        >>> emails = load_emails("emails")
        >>> list(emails.keys())
        ['leave_email.txt', 'project_allocation.txt', ...]
    """
    if not os.path.isdir(email_folder):
        raise FileNotFoundError(
            f"Email folder '{email_folder}' not found. Create it and add "
            f".txt sample emails before running ingest.py."
        )

    emails: dict[str, str] = {}
    for filename in sorted(os.listdir(email_folder)):
        if not filename.endswith(".txt"):
            continue
        file_path = os.path.join(email_folder, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            emails[filename] = f.read()

    logger.info("Loaded %d email files from '%s'", len(emails), email_folder)
    return emails


def split_into_chunks(
    text: str, source: str, chunk_size: int, chunk_overlap: int
) -> List[Chunk]:
    """
    Split a single document into overlapping character-based chunks.

    WHAT IS CHUNKING AND WHY DO WE NEED IT?
    Embedding models and LLMs both have a limited "attention span" (context
    window), and retrieval quality is better when each vector represents a
    focused, single-topic slice of text rather than an entire long
    document. Chunking breaks documents into smaller pieces so that:
      - Each chunk can be embedded and compared independently.
      - Retrieval returns the *specific* relevant passage, not an entire
        document that may only be 10% relevant.

    Our sample emails are short, so in practice most of them fit inside a
    single chunk — but the logic below works correctly for longer
    documents too, which matters once you plug in real historical emails.

    The `chunk_overlap` parameter makes consecutive chunks share a few
    trailing/leading characters, so a sentence that happens to fall right
    on a chunk boundary isn't split away from its surrounding context.

    Parameters:
        text (str): full text of the document.
        source (str): filename this text came from (for traceability).
        chunk_size (int): max characters per chunk.
        chunk_overlap (int): overlapping characters between chunks.

    Returns:
        List[Chunk]: list of chunk dicts with text/source/chunk_id.

    Example:
        >>> split_into_chunks("Hello world" * 100, "demo.txt", 50, 10)
        [{'text': '...', 'source': 'demo.txt', 'chunk_id': 0}, ...]
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [{"text": text, "source": source, "chunk_id": 0}]

    chunks: List[Chunk] = []
    start = 0
    chunk_id = 0
    step = max(chunk_size - chunk_overlap, 1)

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                {"text": chunk_text, "source": source, "chunk_id": chunk_id}
            )
            chunk_id += 1
        start += step

    return chunks


def build_index(
    chunks: List[Chunk], embedding_model_name: str
) -> tuple[faiss.Index, np.ndarray]:
    """
    Embed every chunk and build a FAISS similarity index over the vectors.

    WHAT ARE EMBEDDINGS?
    An embedding is a fixed-length list of numbers (a vector) that
    represents the *meaning* of a piece of text. Texts with similar
    meaning end up with vectors that point in similar directions in this
    high-dimensional space. This is what lets us search by *meaning*
    instead of by exact keyword matching.

    WHAT IS FAISS?
    FAISS (Facebook AI Similarity Search) is a library for efficiently
    searching through large collections of vectors to find the ones most
    similar to a query vector. For a knowledge base of thousands or
    millions of chunks, brute-force comparison would be slow; FAISS uses
    optimized data structures to make this fast.

    WHY COSINE SIMILARITY (via normalized inner product)?
    Cosine similarity measures the *angle* between two vectors rather than
    their raw magnitude, which makes it robust to differences in text
    length. We achieve this with FAISS by L2-normalizing every vector and
    then using an inner-product index — the inner product of two
    normalized vectors is mathematically equivalent to their cosine
    similarity.

    Parameters:
        chunks (List[Chunk]): all chunks to embed and index.
        embedding_model_name (str): Sentence-Transformers model id.

    Returns:
        tuple[faiss.Index, np.ndarray]: the built FAISS index, and the
            raw embedding matrix (returned mainly for logging/debugging).

    Example:
        >>> index, vectors = build_index(chunks, "all-MiniLM-L6-v2")
    """
    logger.info("Loading embedding model '%s'...", embedding_model_name)
    embedder = SentenceTransformer(embedding_model_name)

    texts = [c["text"] for c in chunks]
    logger.info("Generating embeddings for %d chunks...", len(texts))
    vectors = embedder.encode(
        texts, convert_to_numpy=True, show_progress_bar=False
    ).astype("float32")

    # Normalize each vector to unit length so inner product == cosine similarity.
    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)  # IP = Inner Product
    index.add(vectors)

    logger.info(
        "Built FAISS index with %d vectors of dimension %d", index.ntotal, dimension
    )
    return index, vectors


def save_index(index: faiss.Index, chunks: List[Chunk], vector_db_path: str) -> None:
    """
    Persist the FAISS index and chunk metadata to disk.

    We store two files:
      - index.faiss    : the raw FAISS index (binary format understood by faiss)
      - metadata.json   : a JSON list mapping each vector's row position to
                           its original text and source filename

    The metadata file is what lets us go from "vector at row 7 matched
    your query" back to "here is the actual email text and which file it
    came from."

    Parameters:
        index (faiss.Index): the built FAISS index.
        chunks (List[Chunk]): chunk metadata in the same order they were
            added to the index.
        vector_db_path (str): output directory.

    Returns:
        None
    """
    os.makedirs(vector_db_path, exist_ok=True)

    index_path = os.path.join(vector_db_path, "index.faiss")
    metadata_path = os.path.join(vector_db_path, "metadata.json")

    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    logger.info("Saved FAISS index to '%s'", index_path)
    logger.info("Saved metadata to '%s'", metadata_path)


def main() -> None:
    """
    Run the full indexing pipeline end-to-end:
    load emails -> chunk -> embed -> build FAISS index -> save to disk.
    """
    try:
        emails = load_emails(SETTINGS.email_folder)
        if not emails:
            logger.error(
                "No .txt files found in '%s'. Add sample emails and re-run.",
                SETTINGS.email_folder,
            )
            return

        all_chunks: List[Chunk] = []
        for filename, text in emails.items():
            chunks = split_into_chunks(
                text, filename, SETTINGS.chunk_size, SETTINGS.chunk_overlap
            )
            all_chunks.extend(chunks)

        logger.info("Total chunks created: %d", len(all_chunks))

        index, _vectors = build_index(all_chunks, SETTINGS.embedding_model)
        save_index(index, all_chunks, SETTINGS.vector_db_path)

        logger.info("Indexing complete. You can now run app.py")
    except Exception as exc:  # noqa: BLE001
        logger.error("Ingestion failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
