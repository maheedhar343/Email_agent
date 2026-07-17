# AI Email Assistant — Version 1 (RAG-Powered)

## 1. Project Overview

Version 1 upgrades the assistant from Version 0's "ask the LLM directly"
approach to a full **Retrieval-Augmented Generation (RAG)** pipeline.
Instead of relying only on what the LLM memorized during training, the
assistant now:

1. Searches a local knowledge base of real sample emails for the ones
   most similar in meaning to your request.
2. Feeds those examples to the LLM as grounding context.
3. Asks the LLM to write a new email guided by that context.

This produces emails that are more consistent in tone and structure with
*your* reference material, not just generically "LLM-sounding" text.

> **Note on PhiData vs. Agno:** This project was originally scoped around
> "PhiData." PhiData was rebranded to **Agno** in January 2025 and is now
> the actively maintained package (`pip install agno`, `from agno.agent
> import Agent`). All code here uses Agno. The `PHI_API_KEY` variable in
> `.env.example` is kept for reference/back-compat but is not required —
> Agno only needs your `GROQ_API_KEY` for this project's local usage.

**Future versions** (see section 8) will add conversation memory, agentic
decision-making, a web UI, real Gmail sending, and a multi-agent workflow.

## 2. Folder Structure

| Path | Purpose |
|---|---|
| `app.py` | Main application — orchestrates retrieval + generation, runs the terminal loop |
| `rag.py` | Retrieves the Top-K most relevant emails from the FAISS index for a given query |
| `ingest.py` | One-time script that builds the FAISS index from `emails/` |
| `prompts.py` | All prompt templates (system instructions, context formatting, output format) |
| `config.py` | Centralized configuration (model names, paths, chunk size, top-K) |
| `emails/` | The knowledge base — sample business emails in plain text |
| `vector_store/` | Generated FAISS index + metadata (created by `ingest.py`) |
| `.env.example` | Template for secrets/config — copy to `.env` |
| `requirements.txt` | Python dependencies |

## 3. Installation

### 3.1 Virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\Activate.ps1     # Windows PowerShell
```

### 3.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 3.3 API Key Setup

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=gsk_your_real_key
MODEL_NAME=llama-3.3-70b-versatile
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DB_PATH=vector_store
EMAIL_FOLDER=emails
TOP_K=3
```

## 4. Running the Project

**Step 1 — Build the index (run once, and again whenever `emails/` changes):**

```bash
python ingest.py
```

You should see log output confirming how many emails were loaded, how
many chunks were created, and that `vector_store/index.faiss` and
`vector_store/metadata.json` were saved.

**Step 2 — Run the assistant:**

```bash
python app.py
```

Example:

```
You> I need to ask for time off for a family event

----------------------------------------------------------------------
Subject: Leave Request for Family Event

Dear [Manager's Name],
...
----------------------------------------------------------------------
```

Type `exit` to quit.

## 5. Complete RAG Flow

```
User Prompt
    |
    v
Embed the query (SentenceTransformer)
    |
    v
Search FAISS index (cosine similarity)
    |
    v
Top-K most similar email chunks + source files
    |
    v
Prompt Builder (prompts.py) combines:
    - system instructions
    - retrieved context
    - user's request
    - required output format
    |
    v
Groq LLM (via Agno Agent)
    |
    v
Generated Email (printed to terminal)
```

## 6. Explain Every File

### `config.py`
- **Purpose:** single source of truth for every tunable setting.
- **When it runs:** imported by every other module at startup.
- **Input:** `.env` file + hard-coded defaults.
- **Output:** a `SETTINGS` object other files import.
- **Internal workflow:** loads `.env` via `python-dotenv`, then builds an
  immutable `Settings` dataclass.

### `ingest.py`
- **Purpose:** builds the FAISS knowledge base from `emails/`.
- **When it runs:** manually, once, before first use, and again whenever
  emails are added/edited/removed.
- **Input:** `.txt` files in `emails/`.
- **Output:** `vector_store/index.faiss`, `vector_store/metadata.json`.
- **Internal workflow:** load emails → chunk → embed → build FAISS index → save.

### `rag.py`
- **Purpose:** answers "what's most similar to this query?" against the
  saved index.
- **When it runs:** every time `app.py` handles a user request.
- **Input:** a text query string.
- **Output:** a ranked list of `{text, source, score}` dicts.
- **Internal workflow:** embed the query with the same model used in
  `ingest.py` → normalize → FAISS search → map result rows back to text
  via `metadata.json`.

### `prompts.py`
- **Purpose:** assembles the final text sent to the LLM.
- **When it runs:** every request, after retrieval, before the LLM call.
- **Input:** user request string + retrieved chunks.
- **Output:** one combined prompt string.
- **Internal workflow:** formats retrieved chunks into a readable block,
  then concatenates system instructions + context + request + output
  format spec.

### `app.py`
- **Purpose:** ties everything together and runs the interactive loop.
- **When it runs:** `python app.py`.
- **Input:** terminal text from the user.
- **Output:** printed email text.
- **Internal workflow:** build agent → build retriever → loop: read input
  → retrieve → build prompt → call LLM → print result.

## 7. Explain Every Function

Every function across `config.py`, `ingest.py`, `rag.py`, `prompts.py`,
and `app.py` has a docstring in the source code covering its **purpose,
parameters, return value, and a usage example** — open any file and read
top-to-bottom for the full reference. Key functions to start with:

- `ingest.split_into_chunks()` — the chunking logic
- `ingest.build_index()` — the embedding + FAISS index construction
- `rag.Retriever.retrieve()` — the query-time similarity search
- `prompts.build_email_prompt()` — final prompt assembly
- `app.generate_email()` — the full pipeline for one request

## 8. RAG Concepts Explained

**What are embeddings?**
A numeric vector representation of text, where texts with similar meaning
produce vectors that are close together in vector space. This lets us
compare meaning mathematically instead of matching exact words.

**What is chunking?**
Splitting long documents into smaller, focused pieces before embedding
them, so retrieval can return a specific relevant passage instead of an
entire loosely-related document. See the detailed explanation in
`ingest.split_into_chunks()`.

**What is FAISS?**
A library (from Meta AI) for fast similarity search over large
collections of vectors. It lets us find the "nearest neighbors" to a
query vector without brute-force comparing it to every single stored
vector one by one as the dataset grows large.

**Why cosine similarity?**
Cosine similarity compares the *direction* (meaning) of two vectors
rather than their length, making comparisons robust to differences in
text length. We implement it by normalizing vectors to unit length and
using FAISS's inner-product index — inner product of two normalized
vectors equals their cosine similarity.

**What is retrieval?**
The process of searching a knowledge base for the pieces of content most
relevant to a given query — the "R" in RAG.

**What is augmentation?**
Inserting the retrieved content into the LLM's prompt as extra context —
"augmenting" the model's own knowledge with information it wouldn't
otherwise have (e.g. your specific email examples).

**Why does RAG improve responses?**
Because the LLM is grounded in real, relevant, specific examples at
generation time rather than relying solely on generic patterns learned
during training. This produces output that's more consistent with your
actual style/context and reduces made-up ("hallucinated") specifics.

## 9. Learning Notes

- **After `config.py`:** you should understand why centralizing
  configuration avoids "magic numbers" and makes systems easier to tune.
- **After `ingest.py`:** you should understand the difference between the
  one-time "build" phase and the fast "query" phase of a RAG system, and
  be able to explain chunking and embeddings in your own words.
- **After `rag.py`:** you should understand how a query gets turned into
  a vector and compared against stored vectors to find relevant results.
- **After `prompts.py`:** you should understand why prompt construction
  is treated as its own concern, separate from retrieval and generation.
- **After `app.py`:** you should be able to trace a single user request
  all the way through the full RAG pipeline, end to end.

## 10. Future Versions

- **Version 2 — Conversation Memory:** the assistant remembers prior
  turns in a session (e.g. "make it more formal" refers to the last
  email generated), using Agno's built-in session/memory support.
- **Version 3 — Agentic Email Assistant:** the agent decides for itself
  when to retrieve, when to ask a clarifying question, and when it has
  enough information to draft — rather than always retrieving on a fixed
  schedule.
- **Version 4 — Web Interface:** wrap the pipeline in a small web UI
  (e.g. Streamlit or FastAPI + frontend) instead of the terminal.
- **Version 5 — Gmail Integration:** connect to the Gmail API to save
  generated emails as real drafts, or read past sent mail to grow the
  knowledge base automatically.
- **Version 6 — Multi-Agent Email Workflow:** split responsibilities
  across multiple cooperating agents — e.g. a drafting agent, a
  tone/clarity reviewing agent, and a fact-checking agent — coordinated
  by Agno's multi-agent orchestration.
