#AI Medical Assistant Chatbot

A Retrieval-Augmented Generation (RAG) powered medical assistant chatbot built with LangChain, OpenAI, FAISS, and Gradio.

---

## Project Structure

```
RAG_AI_application/
Agentic_ai_with_RAG.py     # Builds and saves the FAISS vector store
gradio_chatbot.py          # Gradio UI + RAG chain (main entry point)
ai-medical-chatbot.txt     # Medical knowledge base (source dataset)
faiss_doc_idx/             # Saved FAISS vector index (auto-generated)
.env                       # API keys (never commit this)
requirements.txt           # Pinned dependencies
README.md
```

---

## How It Works

1. **Ingestion** - Reads `ai-medical-chatbot.txt`, cleans and chunks the text, generates OpenAI embeddings, and saves a FAISS vector index to disk.
2. **Retrieval** - On each user query, finds the top 4 most similar chunks from the FAISS index.
3. **Generation** - Passes retrieved chunks + question to `gpt-3.5-turbo` to generate a grounded answer.
4. **UI** - Gradio 5 ChatInterface provides a simple chat interface.

---

## Setup

### 1. Create a Virtual Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## Running the App

```bash
python3.11 gradio_chatbot.py
```

This will rebuild the FAISS vector store and launch the Gradio UI at `http://localhost:7860`.

To only rebuild the vector store without launching the UI:

```bash
python3.11 Agentic_ai_with_RAG.py
```

---

## Dependencies

| Package                  | Version  | Purpose                                   |
| ------------------------ | -------- | ----------------------------------------- |
| langchain                | 0.2.17   | Core orchestration                        |
| langchain-core           | 0.2.43   | Prompts, runnables, LCEL                  |
| langchain-openai         | 0.1.25   | OpenAI LLM and Embeddings                 |
| langchain-community      | 0.2.17   | FAISS vector store integration            |
| langchain-text-splitters | 0.2.4    | Text chunking                             |
| faiss-cpu                | 1.7.4    | Vector similarity search                  |
| openai                   | >=1.56.1 | OpenAI API client                         |
| httpx                    | 0.27.2   | HTTP client (pinned to avoid proxies bug) |
| gradio                   | >=5.0.0  | Chat UI                                   |
| jinja2                   | >=3.1.4  | Gradio template rendering                 |
| python-dotenv            | 1.0.1    | .env file loading                         |
| tiktoken                 | 0.8.0    | Token counting                            |

---

## Configuration

### Change the LLM model

In `gradio_chatbot.py`:

```python
llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0.2)
```

### Change chunk size

In `Agentic_ai_with_RAG.py`:

```python
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=200)
```

### Change number of retrieved chunks

In `gradio_chatbot.py`:

```python
retriever = vectorStore.as_retriever(search_kwargs={"k": 6})
```

---

## Known Issues

| Error                                        | Cause                        | Fix                                   |
| -------------------------------------------- | ---------------------------- | ------------------------------------- |
| ModuleNotFoundError: langchain.text_splitter | LangChain 0.3+ moved modules | Use langchain_text_splitters          |
| ModuleNotFoundError: langchain.prompts       | Module moved                 | Use langchain_core.prompts            |
| ModuleNotFoundError: langchain.chains        | RetrievalQA removed in 0.3+  | Use LCEL chain                        |
| unexpected keyword argument 'proxies'        | httpx>=0.28 removed proxies  | Pin httpx==0.27.2                     |
| TypeError: unhashable type dict              | Gradio 4.x + Jinja2 conflict | Upgrade to gradio>=5.0.0              |
| unexpected keyword argument 'retry_btn'      | Removed in Gradio 5          | Remove retry_btn, undo_btn, clear_btn |

---

## Security Notes

- Never commit your `.env` file.
- Only load FAISS indexes you generated yourself (`allow_dangerous_deserialization=True`).
- Add the following to `.gitignore`:

```
.env
faiss_doc_idx/
.venv/
__pycache__/
```

---

## License

MIT License - free to use, modify, and distribute.
