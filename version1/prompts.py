"""
prompts.py — Prompt Templates
================================

WHY PROMPTS LIVE IN THEIR OWN FILE
------------------------------------
Prompt text is "code" too — it directly determines output quality — but it
is also the piece you'll want to iterate on most often while learning. By
keeping every template in one file:

- You can tweak wording without touching retrieval or application logic.
- It's obvious, at a glance, exactly what the LLM sees on every call.
- Different prompt strategies (few-shot, chain-of-thought, strict format)
  can be swapped in without changing app.py.

This file exposes a single function, `build_email_prompt()`, which
assembles the four required pieces of a good RAG prompt:
  1. System instructions (how to behave)
  2. Retrieved context (grounding examples from the knowledge base)
  3. The user's actual request
  4. The expected output format
"""

from __future__ import annotations

from typing import List

from rag import RetrievedChunk

# --------------------------------------------------------------------------
# System instructions — sets the assistant's role and ground rules.
# --------------------------------------------------------------------------
SYSTEM_INSTRUCTIONS = """You are a professional email-writing assistant.

You will be given:
1. A set of REFERENCE EMAILS retrieved from a knowledge base of real,
   well-written business emails.
2. A REQUEST describing the email the user wants written.

Your job is to write a new, original email that fulfills the REQUEST,
using the REFERENCE EMAILS only as style and structure guidance (tone,
formatting, level of formality, typical phrasing). Do NOT copy sentences
verbatim from the reference emails unless they are generic placeholders
like [Your Name]. Do NOT mention that you were given reference emails.
"""

# --------------------------------------------------------------------------
# Expected output format — kept separate so it's easy to change independent
# of the system instructions above.
# --------------------------------------------------------------------------
OUTPUT_FORMAT_INSTRUCTIONS = """Respond using exactly this structure and
nothing else (no preamble, no explanation):

Subject: <short, clear subject line>

<Greeting>

<Body — 2 to 4 short, professional paragraphs>

<Closing line>

<Signature placeholder, e.g. "Best regards,\\n[Your Name]">
"""


def format_retrieved_context(chunks: List[RetrievedChunk]) -> str:
    """
    Turn a list of retrieved chunks into a readable block of text to embed
    inside the LLM prompt.

    Parameters:
        chunks (List[RetrievedChunk]): output of Retriever.retrieve().

    Returns:
        str: formatted context block. Returns a placeholder message if no
            chunks were retrieved (so the prompt still makes sense).

    Example:
        >>> format_retrieved_context([{"text": "...", "source": "a.txt", "score": 0.8}])
        'Reference Email 1 (from a.txt):\\n...\\n'
    """
    if not chunks:
        return "(No closely matching reference emails were found.)"

    formatted_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        formatted_blocks.append(
            f"Reference Email {i} (source: {chunk['source']}, "
            f"similarity: {chunk['score']:.2f}):\n{chunk['text']}"
        )
    return "\n\n".join(formatted_blocks)


def build_email_prompt(user_request: str, retrieved_chunks: List[RetrievedChunk]) -> str:
    """
    Assemble the final prompt sent to the LLM, combining system
    instructions, retrieved context, the user's request, and the required
    output format.

    Parameters:
        user_request (str): the user's raw instruction, e.g.
            "Write an email requesting project allocation."
        retrieved_chunks (List[RetrievedChunk]): Top-K similar emails
            retrieved from the knowledge base.

    Returns:
        str: the complete prompt text ready to send to the LLM.

    Example:
        >>> prompt = build_email_prompt("Ask for a raise", retrieved_chunks)
    """
    context_block = format_retrieved_context(retrieved_chunks)

    return f"""{SYSTEM_INSTRUCTIONS}

--- REFERENCE EMAILS ---
{context_block}

--- REQUEST ---
{user_request}

--- OUTPUT FORMAT ---
{OUTPUT_FORMAT_INSTRUCTIONS}
"""
