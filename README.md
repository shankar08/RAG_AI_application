# ðŸ©º Neil â€” AI Medical Assistant Chatbot

A Retrieval-Augmented Generation (RAG) powered medical assistant chatbot built with LangChain, OpenAI, FAISS, and Gradio. Neil answers patient questions based strictly on a curated medical knowledge base, without hallucinating or guessing beyond the provided context.

---

## ðŸ“ Project Structure

```
RAG_AI_application/
â”œâ”€â”€ Agentic_ai_with_RAG.py     # Builds and saves the FAISS vector store
â”œâ”€â”€ gradio_chatbot.py          # Gradio UI + RAG chain (main entry point)
â”œâ”€â”€ ai-medical-chatbot.txt     # Medical knowledge base (source dataset)
â”œâ”€â”€ faiss_doc_idx/             # Saved FAISS vector index (auto-generated)
â”‚   â”œâ”€â”€ index.faiss
â”‚   â””â”€â”€ index.pkl
â”œâ”€â”€ .env                       # API keys (never commit this)
â”œâ”€â”€ requirements.txt           # Pinned dependencies
â””â”€â”€ README.md
```

---

## ðŸ§ How It Works

```
ai-medical-chatbot.txt
        â”‚
        â–¼
  Text Cleaning & Parsing
        â”‚
        â–¼
  CharacterTextSplitter (chunk_size=300, overlap=128)
        â”‚
        â–¼
  OpenAIEmbeddings â†’ FAISS Vector Store
        â”‚
        â–¼
  User Query â†’ Retriever (top-k=4 similar chunks)
        â”‚
        â–¼
  PromptTemplate + ChatOpenAI (gpt-3.5-turbo)
        â”‚
        â–¼
  Answer â†’ Gradio Chat UI
```

1. **Ingestion** (`Agentic_ai_with_RAG.py`) â€” Reads the medical text file, cleans and parses it into question-answer pairs, chunks the content, generates OpenAI embeddings, and saves a FAISS vector index to disk.
2. **Retrieval** â€” On each user query, the retriever finds the top 4 most semantically similar chunks from the FAISS index.
3. **Generation** â€” The retrieved chunks are injected into a prompt template and passed to `gpt-3.5-turbo` to generate a grounded answer.
4. **UI** â€” Gradio 5 `ChatInterface` provides a clean chat UI with example questions.

---

## âš™ï¸ Setup

### 1. Clone / Download the Project

```bash
cd "RAG_AI_application"
```

### 2. Create a Virtual Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Your OpenAI API Key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> âš ï¸ Never commit `.env` to Git. Add it to `.gitignore`.

---

## ðŸš€ Running the App

Simply run the Gradio chatbot â€” it automatically rebuilds the FAISS index on every startup:

```bash
python3.11 gradio_chatbot.py
```

This will:

1. Rebuild the FAISS vector store from `ai-medical-chatbot.txt`
2. Launch the Gradio UI at `http://localhost:7860`
3. Optionally generate a public share link (expires in 72 hours)

### Run the Ingestion Script Standalone (Optional)

If you only want to rebuild the vector store without launching the UI:

```bash
python3.11 Agentic_ai_with_RAG.py
```

---

## ðŸ“¦ Dependencies

| Package                    | Version  | Purpose                                   |
| -------------------------- | -------- | ----------------------------------------- |
| `langchain`                | 0.2.17   | Core orchestration                        |
| `langchain-core`           | 0.2.43   | Prompts, runnables, LCEL                  |
| `langchain-openai`         | 0.1.25   | OpenAI LLM + Embeddings                   |
| `langchain-community`      | 0.2.17   | FAISS vector store integration            |
| `langchain-text-splitters` | 0.2.4    | Text chunking                             |
| `faiss-cpu`                | 1.7.4    | Vector similarity search                  |
| `openai`                   | >=1.56.1 | OpenAI API client                         |
| `httpx`                    | 0.27.2   | HTTP client (pinned to avoid proxies bug) |
| `gradio`                   | >=5.0.0  | Chat UI                                   |
| `jinja2`                   | >=3.1.4  | Gradio template rendering                 |
| `python-dotenv`            | 1.0.1    | `.env` file loading                       |
| `tiktoken`                 | 0.8.0    | Token counting                            |

---

## ðŸ”§ Configuration

### Changing the LLM Model

In `gradio_chatbot.py`, update the `ChatOpenAI` model name:

```python
llm = ChatOpenAI(
    model="gpt-4o",       # upgrade to GPT-4o for better answers
    api_key=api_key,
    temperature=0.2
)
```

### Changing Chunk Size

In `Agentic_ai_with_RAG.py`, adjust the splitter parameters:

```python
text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=500,       # increase for more context per chunk
    chunk_overlap=200,    # increase overlap for better continuity
    length_function=len
)
```

### Changing Number of Retrieved Chunks

In `gradio_chatbot.py`, update the retriever:

```python
retriever = vectorStore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}    # retrieve 6 chunks instead of 4
)
```

---

## âš ï¸ Known Issues & Fixes

| Error                                          | Cause                         | Fix                                         |
| ---------------------------------------------- | ----------------------------- | ------------------------------------------- |
| `ModuleNotFoundError: langchain.text_splitter` | LangChain 0.3+ moved modules  | Use `langchain_text_splitters`              |
| `ModuleNotFoundError: langchain.prompts`       | Module moved                  | Use `langchain_core.prompts`                |
| `ModuleNotFoundError: langchain.chains`        | `RetrievalQA` removed in 0.3+ | Use LCEL chain                              |
| `unexpected keyword argument 'proxies'`        | `httpx>=0.28` removed proxies | Pin `httpx==0.27.2`                         |
| `TypeError: unhashable type: 'dict'`           | Gradio 4.x + Jinja2 conflict  | Upgrade to `gradio>=5.0.0`                  |
| `unexpected keyword argument 'retry_btn'`      | Removed in Gradio 5           | Remove `retry_btn`, `undo_btn`, `clear_btn` |

---

## ðŸ”’ Security Notes

- Never commit your `.env` file or expose your `OPENAI_API_KEY`.
- The `allow_dangerous_deserialization=True` flag is required to load FAISS indexes from disk â€” only load indexes you generated yourself.
- Add `.env` and `faiss_doc_idx/` to `.gitignore`:

```gitignore
.env
faiss_doc_idx/
.venv/
__pycache__/
*.pyc
```

---

## ðŸ“„ License

MIT License â€” free to use, modify, and distribute.
