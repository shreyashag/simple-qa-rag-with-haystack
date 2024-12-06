# Retrieval-Augmented QA Pipeline with Haystack and Ollama

This repository contains an example of building a Retrieval-Augmented Generation (RAG) pipeline using [Haystack](https://github.com/deepset-ai/haystack) and [Ollama](https://docs.ollama.ai/). The pipeline retrieves relevant documents from a dataset and uses a local LLM (provided by Ollama) to generate context-aware answers without relying on external APIs.

## Features

- **RAG Approach:** Combines an LLM with retrieved documents to produce more accurate, contextually grounded answers.
- **Local Inference with Ollama:** Run models locally to avoid expensive API calls and maintain privacy.
- **Simple Setup:** Uses an in-memory DocumentStore and Sentence Transformers for embedding queries and documents.
- **Adaptable:** Easily switch out datasets, models, or vector databases as needed.

## Prerequisites

- **Python 3.10+**
- **Pip** and basic knowledge of Python environments
- **Ollama** installed and running locally. For installation instructions, see the [Ollama documentation](https://docs.ollama.ai/getting-started/installation).

## Quick Start

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/shreyashag/simple-qa-rag-with-haystack.git
   cd simple-qa-rag-with-haystack
   ```

2. **Create a Virtual Environment (Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   
   On Windows:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Set Environment Variables:**
   Create a `.env` file in the project root, follow the `.env.sample` for keys:
   
   ```bash
   echo 'OLLAMA_ENDPOINT="http://localhost:11434"' >> .env
   echo 'OLLAMA_MODEL="qwen2.5-coder"' >> .env
   ```
   
   Make sure Ollama is running. For details, see the [Ollama Documentation](https://docs.ollama.ai/).

5. **Run the Script:**
   ```bash
   python qa_pipeline_with_retrieval_augmentation.py
   ```
   
   You should see answers printed to the console for sample questions.

## How It Works

1. **Data Loading:**  
   The code fetches the "Seven Wonders of the Ancient World" dataset from Hugging Face and wraps them into Haystack `Document` objects.

2. **Embedding and Indexing:**  
   Using `SentenceTransformersDocumentEmbedder`, we create embeddings for each document and store them in an `InMemoryDocumentStore`.

3. **Pipeline Construction:**
   - **TextEmbedder:** Converts user queries into embeddings.
   - **Retriever:** Uses embeddings to find relevant documents.
   - **PromptBuilder:** Constructs a prompt that includes both the user’s query and the retrieved documents.
   - **OllamaGenerator:** Passes the prompt to Ollama’s locally running LLM to generate the final answer.

4. **Querying:**
   By feeding a question through the pipeline, you get a fact-based response that references the provided documents.

## Customization

- **Change the Model:** Update the `OLLAMA_MODEL` in the `.env` file to try different models.
- **Use Another Document Store:** Replace `InMemoryDocumentStore` with a vector database like FAISS or Weaviate for larger-scale projects.
- **Adjust Prompts:** Modify the prompt template in the code to alter the behavior and style of generated responses.

## Troubleshooting

- Ensure Ollama is running and accessible at the URL in `.env`.
- Make sure all Python dependencies are installed and that you’re using a supported Python version.
- If embedding fails, confirm that `sentence-transformers` models are downloading correctly and your internet connection is stable.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.