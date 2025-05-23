# Llama Index RAG

A self-contained Retrieval-Augmented Generation (RAG) pipeline using [LlamaIndex](https://github.com/jerryjliu/llama_index) and Azure OpenAI, designed for document ingestion, indexing, and semantic querying. This project demonstrates how to process PDF documents, extract structured information, build a vector index (using FAISS), and perform advanced question answering over your data.

---

## Features

- **PDF Ingestion**: Load and parse PDF documents for downstream processing.
- **Document Processing Pipeline**: Extract summaries, questions answered, and other metadata using LlamaIndex extractors.
- **Vector Indexing**: Store document embeddings in a FAISS vector store for efficient similarity search.
- **Azure OpenAI Integration**: Use Azure OpenAI for both LLM and embedding models.
- **Semantic Querying**: Query your indexed documents using natural language and get detailed, context-aware answers.
- **Modular Design**: Easily extend or adapt the pipeline for new document types or vector stores.

---

## Project Structure

```
Llama Index RAG/
│
├── app.py                # Main pipeline: ingest, process, index, and query documents
├── query.py              # Query-only interface for existing indexes
├── services/
│   └── llm.py            # Azure OpenAI LLM and embedding model initialization
├── input/
│   └── Attention is all you need.pdf   # Example PDF document
├── storage/              # Persisted FAISS index and metadata
├── config.json           # Configuration for models and parameters
├── .env                  # Environment variables for Azure OpenAI credentials
└── ReadMe.md             # Project documentation
```

---

## Setup

1. **Clone the repository**

   ```sh
   git clone <repo-url>
   cd Llama\ Index\ RAG
   ```

2. **Install dependencies**

   ```sh
   pip install -r requirements.txt
   ```

3. **Configure Azure OpenAI credentials**

   Create a `.env` file in the project root:

   ```
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_ENDPOINT=your_endpoint
   AZURE_OPENAI_API_VERSION=your_api_version
   AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
   AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=your_embedding_deployment_name
   AZURE_OPENAI_EMBEDDING_API_KEY=your_embedding_api_key
   AZURE_OPENAI_EMBEDDING_ENDPOINT=your_embedding_endpoint
   AZURE_OPENAI_EMBEDDING_API_VERSION=your_embedding_api_version
   ```

4. **Edit `config.json`** (if needed) to set model names, temperature, etc.

---

## Usage

### 1. Ingest and Index a PDF

```sh
python app.py
```

- Loads the PDF from `input/Attention is all you need.pdf`
- Processes and indexes the document
- Saves the FAISS index to `./storage`
- Example query is run at the end

### 2. Query an Existing Index

```sh
python query.py
```

- Loads the existing index from `./storage`
- Runs a sample query

---

## Customization

- **Change the input PDF**: Place your PDF in the `input/` folder and update the path in `app.py` or `query.py`.
- **Add more extractors**: Modify `create_processing_pipeline()` in `app.py` to include additional extractors or transformations.
- **Switch vector store**: Swap out FAISS for another supported vector store (e.g., Milvus) by editing the relevant code sections.

---

## Requirements

- Python 3.8+
- [LlamaIndex](https://github.com/jerryjliu/llama_index)
- [FAISS](https://github.com/facebookresearch/faiss)
- Azure OpenAI account and deployed models

---

## License

MIT License

---

## Acknowledgements

- [LlamaIndex](https://github.com/jerryjliu/llama_index)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)