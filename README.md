# 📚 RAG-Powered Document Assistant

A complete Retrieval-Augmented Generation product: raw documents → chunking → embeddings → persisted
vector store → FastAPI service → Streamlit chat UI. Answers are **grounded in the retrieved
passages and cited**; when the documents do not contain the answer, the assistant says so instead of
guessing.

**Track:** Core (text-only RAG) · **LLM:** local Ollama · **Vector DB:** Chroma · **Frontend:** Streamlit

---

## 1. Architecture

```
                  ┌────────────────────────────────────────────────┐
   data/raw/ ───▶ │  notebooks/rag_pipeline.ipynb                  │
   (PDF, TXT)     │  load → clean → chunk (800/150) → embed        │
                  │  (all-MiniLM-L6-v2) → Chroma → evaluate        │
                  └──────────────────────┬─────────────────────────┘
                                         │ 2.7 export (no rebuild at request time)
                                         ▼
                            backend/data/vector_store/
                                         │
┌──────────────┐   HTTP POST /query      ▼                    ┌──────────────────┐
│  Streamlit   │ ───────────────▶ ┌─────────────────┐         │  Ollama daemon   │
│ frontend/    │                  │  FastAPI        │ ──────▶ │  llama3.2:3b     │
│ app.py       │ ◀─────────────── │  backend/       │ ◀────── │  :11434          │
└──────────────┘  answer+sources  │  retrieval +    │         └──────────────────┘
                                  │  generation     │
                                  └─────────────────┘
```

**Request flow:** question → embed → Chroma top-k (cosine) → relevance floor → numbered prompt with
sources → Ollama → answer with `[n]` citations → JSON `{answer, sources, contexts, grounded}` →
rendered in the chat UI with an expandable source panel.

## 2. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Parsing | `pypdf` | Pure-python page-level text extraction, detects scan-only pages |
| Embeddings | `all-MiniLM-L6-v2` (384-d) | Fast on CPU, strong on short-passage semantic search |
| Vector DB | Chroma (`PersistentClient`, cosine) | Persists to disk, loaded directly by the backend |
| LLM | Ollama `llama3.2:3b`, temperature 0 | Local, free, deterministic for evaluation |
| API | FastAPI + Pydantic v2 | Validation (422 on bad input), automatic Swagger docs |
| UI | Streamlit `st.chat_message` | Chat interface with loading state and source panel |
| Tests | pytest + `TestClient` | Dependency overrides → tests pass without Ollama |

## 3. Project Structure

```
rag-assistant-app/
├── notebooks/rag_pipeline.ipynb      # Phase 2: build + evaluate (Restart & Run All safe)
├── config/eval_questions.json        # 10 evaluation questions (incl. 1 out-of-scope)
├── scripts/fetch_corpus.py           # optional corpus downloader
├── scripts/verify_submission.py      # self-check against the deliverables checklist
├── colab/run_on_colab.ipynb          # one-click Google Colab runner (Ollama + API + Streamlit)
├── data/raw/                         # source documents (git-ignored)
├── reports/                          # evaluation_results.csv / .md (generated)
├── backend/
│   ├── app/
│   │   ├── main.py                   # FastAPI app, CORS, lifespan startup loading
│   │   ├── api/routes/query.py       # GET /health, POST /query
│   │   ├── core/config.py            # settings from .env
│   │   ├── schemas/query.py          # QueryRequest / QueryResponse
│   │   ├── services/retrieval.py     # load vector store, retrieve chunks
│   │   ├── services/generation.py    # prompt + Ollama call
│   │   └── utils/logging_config.py
│   ├── data/vector_store/            # produced by the notebook (git-ignored)
│   ├── tests/test_query.py           # happy path + invalid input (422)
│   ├── requirements.txt · .env.example · Dockerfile
└── frontend/
    ├── app.py · api_client.py · requirements.txt · .env.example
```

## 4. Domain & Data

**Domain:** Machine Learning / NLP Research Papers — foundational and RAG-specific literature (Transformer, BERT, RAG, Self-RAG, RAGAS).

**Corpus:** _N_ PDFs, _M_ pages, ~_X_ chunks. Documents are text-extractable; the loader flags any
file yielding under ~100 characters per page as `needs_ocr?` and excludes it.

The raw corpus is **not committed** (size + licensing). To reproduce it:

```bash
python scripts/fetch_corpus.py      # downloads the default open-access PDF set into data/raw/
```

Or drop your own `.pdf` / `.txt` / `.md` files into `data/raw/` and edit
`config/eval_questions.json` so the 10 evaluation questions match your domain.

## 5. Setup

### 5.0 Prerequisites

```bash
python --version     # >= 3.10
ollama --version
git --version
ollama pull llama3.2:3b
```

### 5.1 Notebook — build the vector store (run this first)

```bash
git clone https://github.com/<your-username>/rag-assistant-app.git
cd rag-assistant-app
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements-notebook.txt
python scripts/fetch_corpus.py                 # or add your own documents to data/raw/
jupyter notebook notebooks/rag_pipeline.ipynb  # Kernel → Restart & Run All
```

Section 2.7 exports the store into `backend/data/vector_store/`.

### 5.2 Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env            # copy .env.example .env  (Windows)
pytest -q                       # 3 passed
uvicorn app.main:app --reload   # http://localhost:8000/docs
```

### 5.3 Frontend

```bash
cd frontend
pip install -r requirements.txt
cp .env.example .env            # API_BASE_URL=http://localhost:8000
streamlit run app.py            # http://localhost:8501
```

Backend on `:8000`, frontend on `:8501` — ask a question and the answer appears with its citations.

### 5.4 Google Colab

No local install: open `colab/run_on_colab.ipynb` in Colab and run it top to bottom. It installs
Ollama, keeps the corpus and vector store in Google Drive, executes the pipeline notebook, starts
the API and Streamlit, and exposes the app over a Cloudflare tunnel. Run the live instructor demo
locally, though — a tunnel dying mid-demo is an avoidable risk.

### 5.5 Docker (backend only)

```bash
cd backend
docker build -t rag-backend .
docker run -p 8000:8000 -e OLLAMA_HOST=http://host.docker.internal:11434 rag-backend
```

## 6. Environment Variables

**backend/.env**

| Variable | Default | Description |
|---|---|---|
| `VECTOR_STORE_DIR` | `data/vector_store` | Persisted Chroma store exported by the notebook |
| `COLLECTION_NAME` | `documents` | Chroma collection name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Must match the model used to build the store |
| `TOP_K` | `4` | Chunks retrieved per question |
| `MIN_RELEVANCE` | `0.15` | Cosine-similarity floor; below it the assistant refuses |
| `MAX_CONTEXT_CHARS` | `6000` | Prompt context budget |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama daemon URL |
| `OLLAMA_MODEL` | `llama3.2:3b` | Local model name |
| `OLLAMA_TIMEOUT` | `120` | Seconds |
| `TEMPERATURE` | `0.0` | Deterministic answers |
| `ALLOWED_ORIGINS` | `http://localhost:8501,...` | CORS allow-list |
| `LOG_LEVEL` | `INFO` | Logging level |

**frontend/.env**

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | Backend base URL (never hard-coded in code) |
| `REQUEST_TIMEOUT` | `180` | Seconds to wait for a generated answer |

## 7. API Reference

Interactive docs: `http://localhost:8000/docs`

### `GET /health`

```json
{"status":"ok","vector_store_loaded":true,"chunks_indexed":412,
 "embedding_model":"sentence-transformers/all-MiniLM-L6-v2",
 "llm_model":"llama3.2:3b","llm_reachable":true}
```

### `POST /query`

Request — `{"question": "string (3–1000 chars)", "top_k": 1-10 (optional)}`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is retrieval-augmented generation?"}'
```

Response:

```json
{
  "answer": "Retrieval-augmented generation combines a retriever over a document corpus with a generator model [1][2].",
  "sources": ["rag_knowledge_intensive_nlp.pdf (p. 2)", "self_rag.pdf (p. 1)"],
  "contexts": [
    {"citation":"rag_knowledge_intensive_nlp.pdf (p. 2)","document":"rag_knowledge_intensive_nlp.pdf",
     "page":2,"chunk_id":"rag_knowledge_intensive_nlp.pdf::p2::c0","score":0.7412,"preview":"..."}
  ],
  "grounded": true,
  "latency_ms": 2841,
  "model": "llama3.2:3b"
}
```

| Status | Meaning |
|---|---|
| `200` | Answer returned |
| `422` | Validation error (empty/too-long question, `top_k` out of range) |
| `502` | LLM call failed |
| `503` | Vector store or LLM not loaded |

## 8. Evaluation Results

Generated by notebook section 2.6 into `reports/evaluation_results.md` 

| Metric | Result |
|---|---|
| Questions tested | 10 |
| Relevant context retrieved | 10/10 |
| Grounded (cited or correctly refused) | 9/10 |
| Fully correct | 5/10 |
| Median latency | 0.6 s |

| # | Question | Retrieved source | Correct? |
|---|---|---|---|
| 1 | what is RAG? | rag_knowledge_intensive_nlp.pdf | Yes |
| 2 | what is BERT? | sentence_bert.pdf | yes |
| 3 | What is the self-attention mechanism in the Transformer architecture? | attention_is_all_you_need.pdf | yes |
| 4 | How does BERT differ from OpenAI GPT in its pre-training approach? | bert.pdf | yes |
| 5 | What are the two pre-training tasks used to train BERT? | bert.pdf | yes |
| 6 | What is the main limitation of standard RAG that Self-RAG attempts to address? | self_rag.pdf | yes |
| 7 | What role do reflection tokens play in Self-RAG | self_rag.pdf | yes |
| 8 | What is the purpose of the RAGAS evaluation framework? | ragas_evaluation.pdf | yes |
| 9 | why does the Transformer use multi-head attention instead of a single attention function? | attention_is_all_you_need.pdf | yes |
| 10 | How does the positional encoding in the original Transformer work? | attention_is_all_you_need.pdf | yes |
| 11 | What is the current population of Tokyo? | none | yes- refused |
| 12 | how are you? | none | Yes — correctly refused |

**Failure cases & mitigations.** Weak-vocabulary questions retrieved loosely related chunks and
tempted the model to answer from pre-training → fixed with a relevance floor plus a mandated refusal
sentence. Answers straddling chunk boundaries were truncated → overlap raised to 150 characters.
Citations were occasionally dropped on long answers → `temperature=0` and an explicit numbered
citation rule in the system prompt. Aggregate questions ("list every method mentioned") remain
weak, since top-k retrieval only sees 4 passages.

## 9. Screenshots

| Streamlit chat | Screenshot `/docs` |
|---|---|
| ![Video](https://drive.google.com/file/d/1h_Cfwb_mUk9OpCVUfkuH_-Ln5CxSP_Q9/view?usp=sharing) | ![Screenshot](https://drive.google.com/file/d/1SuvRu5BKga_X0UIPQnIvC9B_59Ul7o-1/view?usp=sharing) |

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `/health` shows `vector_store_loaded: false` | Run the notebook through section 2.7 |
| `llm_reachable: false` | Start Ollama and `ollama pull llama3.2:3b` |
| Frontend: "API_BASE_URL is not set" | `cp frontend/.env.example frontend/.env` |
| CORS error in the browser | Add the frontend origin to `ALLOWED_ORIGINS` |
| Answers ignore the documents | Lower `TOP_K`, raise `MIN_RELEVANCE`, confirm the store was rebuilt |

## 11. Pre-submission check

```bash
python scripts/verify_submission.py
```

Then clone the repo into a fresh folder and follow **only** this README end to end.
