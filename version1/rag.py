"""
rag.py — Retrieval Pipeline
=============================

WHAT THIS FILE DOES
--------------------
This module loads the FAISS index and metadata that `ingest.py` built, and
exposes a single function — `retrieve()` — that turns a user's query into
the Top-K most similar chunks from the knowledge base.

WHAT IS RETRIEVAL?
Retrieval is the "R" in RAG. Instead of the LLM answering purely from what
it memorized during training, we first *search* our own knowledge base for
passages relevant to the current question, and hand those passages to the
LLM as extra context. This grounds the model's output in real, specific
examples rather than generic training data.

This module is intentionally separate from app.py so it can be reused,
tested, or swapped out (e.g. for a different vector database) without
touching the rest of the application.
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
logger = logging.getLogger("email_assistant.rag")


class RetrievedChunk(TypedDict):
    """A single retrieved result, with its similarity score attached."""
    text: str
    source: str
    score: float


class Retriever:
    """
    Loads a persisted FAISS index + metadata and answers similarity
    queries against it.

    Attributes:
        embedder (SentenceTransformer): model used to embed queries. It
            must be the SAME model that was used in ingest.py, otherwise
            the vector spaces won't be comparable.
        index (faiss.Index): the loaded FAISS index.
        metadata (list[dict]): chunk text/source, aligned by row index
            with vectors in `index`.
    """

    def __init__(self, vector_db_path: str, embedding_model_name: str) -> None:
        """
        Load the FAISS index and metadata from disk.

        Parameters:
            vector_db_path (str): folder containing index.faiss and
                metadata.json (produced by ingest.py).
            embedding_model_name (str): Sentence-Transformers model id;
                must match the one used during ingestion.

        Raises:
            FileNotFoundError: if the index/metadata files don't exist yet
                (i.e. ingest.py hasn't been run).

        Example:
            >>> retriever = Retriever("vector_store", "all-MiniLM-L6-v2")
        """
        index_path = os.path.join(vector_db_path, "index.faiss")
        metadata_path = os.path.join(vector_db_path, "metadata.json")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"No index found at '{vector_db_path}'. "
                f"Run 'python ingest.py' first to build the knowledge base."
            )

        logger.info("Loading FAISS index from '%s'", index_path)
        self.index = faiss.read_index(index_path)

        logger.info("Loading metadata from '%s'", metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata: list[dict] = json.load(f)

        logger.info("Loading embedding model '%s'", embedding_model_name)
        self.embedder = SentenceTransformer(embedding_model_name)

    def retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """
        Find the Top-K chunks most semantically similar to the query.

        Parameters:
            query (str): the user's raw instruction/question, e.g.
                "Write an email requesting project allocation."
            top_k (int): how many results to return.

        Returns:
            List[RetrievedChunk]: results ordered from most to least
                similar, each with its cosine-similarity score.

        Example:
            >>> retriever.retrieve("leave request for a wedding", top_k=2)
            [{'text': '...', 'source': 'leave_email.txt', 'score': 0.71}, ...]
        """
        # Embed the query the same way documents were embedded, then
        # L2-normalize so that inner product == cosine similarity.
        query_vector = self.embedder.encode(
            [query], convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, top_k)

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 when there are fewer than top_k vectors
                continue
            chunk = self.metadata[idx]
            results.append(
                {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "score": float(score),
                }
            )

        logger.info(
            "Retrieved %d chunks for query: '%s'", len(results), query[:60]
        )
        return results


def get_default_retriever() -> Retriever:
    """
    Convenience factory that builds a Retriever using values from config.py.

    Parameters:
        None

    Returns:
        Retriever: ready-to-use retriever instance.
    """
    return Retriever(
        vector_db_path=SETTINGS.vector_db_path,
        embedding_model_name=SETTINGS.embedding_model,
    )


if __name__ == "__main__":
    # Small manual test: `python rag.py` lets you sanity-check retrieval
    # without running the full app.
    retriever = get_default_retriever()
    test_query = "I need to ask my manager for time off"
    for result in retriever.retrieve(test_query, top_k=SETTINGS.top_k):
        print(f"[{result['score']:.3f}] {result['source']}")
        print(result["text"][:120].replace("\n", " ") + "...\n")
