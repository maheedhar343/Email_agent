"""
config.py — Centralized Configuration for Version 1
=====================================================

WHY THIS FILE EXISTS
---------------------
Every tunable value in the RAG pipeline (embedding model name, chunk size,
top-K, file paths, LLM model) lives here, in one place. This means:

- No "magic numbers" scattered across ingest.py / rag.py / app.py.
- Changing the embedding model or top-K for an experiment means editing
  ONE line, not hunting through multiple files.
- Environment-specific values (API keys, paths) are read from .env, while
  algorithm-tuning values (chunk size, top-K) are plain Python constants
  that you're expected to tweak directly in this file as you learn.

This file does not perform any logic — it only loads and exposes values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # populate os.environ from the local .env file


@dataclass(frozen=True)
class Settings:
    """
    Immutable container for all configuration values used across the
    ingestion and retrieval pipeline.

    Attributes:
        groq_api_key (str): Secret key used to call the Groq LLM API.
        model_name (str): Groq-hosted chat model id used for generation.
        embedding_model (str): Sentence-Transformers model id used to
            turn text into vectors.
        email_folder (str): Path to the folder of raw .txt sample emails
            that make up the knowledge base.
        vector_db_path (str): Path to the folder where the FAISS index
            and metadata file are persisted on disk.
        chunk_size (int): Maximum number of characters per chunk when
            splitting long emails before embedding.
        chunk_overlap (int): Number of overlapping characters between
            consecutive chunks, so context isn't lost at chunk boundaries.
        top_k (int): Number of most-similar chunks to retrieve per query.
    """

    groq_api_key: str
    model_name: str
    embedding_model: str
    email_folder: str
    vector_db_path: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


def load_settings() -> Settings:
    """
    Build a Settings object from environment variables plus sensible
    hard-coded defaults for the algorithm-tuning parameters.

    Parameters:
        None

    Returns:
        Settings: fully populated configuration object.

    Example:
        >>> settings = load_settings()
        >>> print(settings.top_k)
        3
    """
    return Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        model_name=os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        email_folder=os.getenv("EMAIL_FOLDER", "emails"),
        vector_db_path=os.getenv("VECTOR_DB_PATH", "vector_store"),
        # --- Algorithm-tuning constants (edit directly to experiment) ---
        chunk_size=800,
        chunk_overlap=100,
        top_k=int(os.getenv("TOP_K", "3")),
    )


# A module-level singleton so every other file can simply do:
#   from config import SETTINGS
SETTINGS = load_settings()
