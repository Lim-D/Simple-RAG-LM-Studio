# Simple-RAG-LM-Studio
A simple python program that creates a chroma vector database and interacts with LM Studio Model Server to achieve a simple RAG pipeline

## Architecture
```
PDF/DOCX/TXT files
        ↓
Text extraction and chunking
        ↓
LM Studio embedding model
        ↓
Persistent Chroma vector database
        ↓
Retrieve 4–8 relevant chunks
        ↓
LM Studio chat model
        ↓
Answer with document citations
```
## How to Use:
### 1. Configure LM Studio

In LM Studio:

1. Download and load an instruction/chat model that fits your computer.
2. Download an embedding model. LM Studio’s documentation uses `nomic-ai/nomic-embed-text-v1.5` as an example. ([LM Studio][3])
3. Open **Developer**.
4. Turn on **Start server**.
5. Leave the default port as `1234`.

### 2. Install and set-up of the project
1. Install 64-bit Python if it is not already installed. Then run `setup.bat`
    - This creates a Python virtual environment, installs the required packages, creates the document folder, and creates your `.env` configuration file.
2. Identify the model names from your LM Studio using `check_lmstudio.bat`
3. Open the generated `.env` file and replace with your model identifiers.

Your resulting configuration should look like this:
```
# LM Studio OpenAI-compatible server
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_API_KEY=lm-studio

# Copy the exact model identifiers shown by: python diagnose.py
CHAT_MODEL=hermes-3-llama-3.2-3b
EMBED_MODEL=text-embedding-nomic-embed-text-v1.5

# Local vector database
CHROMA_PATH=.rag_db
COLLECTION_NAME=local_documents

EMBED_DOCUMENT_PREFIX=search_document:
EMBED_QUERY_PREFIX=search_query:

TOP_K=4
CANDIDATE_MULTIPLIER=4
MIN_RELEVANCE_SCORE=0.45
LEXICAL_WEIGHT=0.12

MAX_CONTEXT_CHARS=12000
CHAT_MAX_TOKENS=900
```

### 3. Add your documents
Copy your files into: `~\documents`

Subfolders are supported:
```
documents\
├── contracts\
│   ├── contract-2024.pdf
│   └── amendments.docx
├── manuals\
│   └── equipment-manual.pdf
└── notes.md
```

### 4. Create the RAG database and Running:
Command: `python ingest.py documents --rebuild --chunk-size 1800 --overlap 250`
Make sure the LM Studio embedding model is loaded while building.

To run the project, just use `run_chat.bat`