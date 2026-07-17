"""
app.py — Version 1: RAG-Powered AI Email Assistant
======================================================

WHAT THIS FILE DOES
--------------------
This is the main application for Version 1. Unlike Version 0 (which asked
the LLM to generate emails from nothing but its own training data), this
version retrieves similar real emails from a local knowledge base first,
and feeds them to the LLM as grounding context — this is
Retrieval-Augmented Generation (RAG).

OVERALL FLOW
------------
    User Prompt
        |
        v
    Retriever (rag.py)  -->  Local Knowledge Base (FAISS index)
        |
        v
    Top-K Similar Emails
        |
        v
    Prompt Builder (prompts.py)
        |
        v
    Groq LLM (via Agno Agent)
        |
        v
    Generated Email

PREREQUISITE
------------
You must run `python ingest.py` once before running this file, so that
`vector_store/index.faiss` and `vector_store/metadata.json` exist.

Run with:
    python app.py
"""

from __future__ import annotations

import logging
import sys

from agno.agent import Agent
from agno.models.groq import Groq

from config import SETTINGS
from prompts import build_email_prompt
from rag import Retriever, get_default_retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("email_assistant.v1")


def build_agent() -> Agent:
    """
    Construct the Agno Agent used to turn a fully-assembled prompt into a
    generated email.

    Unlike Version 0, this agent receives NO system `instructions` of its
    own — the full instruction set (system rules + context + request +
    format) is built manually by prompts.build_email_prompt() and passed
    in as the message on every call. This makes the RAG pipeline's prompt
    construction fully transparent and inspectable, rather than hidden
    inside framework defaults.

    Parameters:
        None

    Returns:
        Agent: a ready-to-use Agno agent configured with the Groq model.

    Example:
        >>> agent = build_agent()
        >>> agent.run("...")
    """
    if not SETTINGS.groq_api_key:
        logger.error(
            "GROQ_API_KEY is missing. Create a .env file (see .env.example) "
            "and set GROQ_API_KEY=<your key>."
        )
        sys.exit(1)

    model = Groq(id=SETTINGS.model_name, api_key=SETTINGS.groq_api_key)
    return Agent(model=model, markdown=False)


def build_retriever() -> Retriever:
    """
    Load the persisted FAISS retriever, with a clear error if `ingest.py`
    hasn't been run yet.

    Parameters:
        None

    Returns:
        Retriever: ready-to-use retriever instance.
    """
    try:
        return get_default_retriever()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)


def generate_email(agent: Agent, retriever: Retriever, user_request: str) -> str:
    """
    Run the full RAG pipeline for a single user request:
    retrieve context -> build prompt -> call LLM -> return generated email.

    Parameters:
        agent (Agent): configured Agno agent (Groq-backed).
        retriever (Retriever): loaded FAISS retriever.
        user_request (str): the user's raw instruction.

    Returns:
        str: the generated email text.

    Raises:
        Exception: re-raised after logging, if the LLM call fails.

    Example:
        >>> generate_email(agent, retriever, "Write a leave request email")
    """
    retrieved_chunks = retriever.retrieve(user_request, top_k=SETTINGS.top_k)

    logger.info(
        "Using %d retrieved chunk(s) as context: %s",
        len(retrieved_chunks),
        [c["source"] for c in retrieved_chunks],
    )

    final_prompt = build_email_prompt(user_request, retrieved_chunks)

    try:
        response = agent.run(final_prompt)
        return response.content
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to generate email: %s", exc)
        raise


def main() -> None:
    """
    Entry point: runs the interactive terminal loop.

    Parameters:
        None

    Returns:
        None
    """
    agent = build_agent()
    retriever = build_retriever()

    print("=" * 70)
    print(" AI Email Assistant — Version 1 (RAG-powered)")
    print(" Type an instruction to generate an email, or 'exit' to quit.")
    print("=" * 70)

    while True:
        try:
            user_request = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_request:
            continue

        if user_request.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            email_text = generate_email(agent, retriever, user_request)
        except Exception:
            print("Something went wrong while generating the email. "
                  "Check the log above for details.")
            continue

        print("\n" + "-" * 70)
        print(email_text)
        print("-" * 70)


if __name__ == "__main__":
    main()
