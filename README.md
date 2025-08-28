# IntraDocAI 

**IntraDocAI** is a **Retrieval-Augmented Generation (RAG) system** that analyzes company or official documents and provides accurate answers to **queries**.  
It combines vector search with LLM reasoning to ensure context-aware, reliable responses. Useful for analysing company documents and agreements in minutes. 

## Features

- **RAG-based Question Answering** – Handles vague or complex queries by grounding responses in the source documents.  
- **Multi-format Document Support** – Works with PDFs, Word docs, and text files.  
- **Semantic Search with Qdrant** – Efficient vector database for storing and retrieving document chunks.  
- **OpenAI API Integration** – Leverages LLMs for natural language understanding and generation.  
- **Accurate & Contextual Answers** – Always backed by relevant document context.  
- **Secure & Scalable** – Designed for enterprise-level document analysis.

## Steps : 
1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/document-insight-ai.git
   cd document-insight-ai
   ``` 
2. ***Set up the virtual environment***
   ``` python 
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Create a .env file in the project root:

      OPENAI_API_KEY=your_openai_api_key
      QDRANT_URL=your_qdrant_instance_url
      QDRANT_API_KEY=your_qdrant_api_key

4. Start the FastAPI Server

   ``` bash
   uvicorn main:app --reload
   ```



- **Framework:** Python + FastAPI  
- **Vector Database:** [Qdrant](https://qdrant.tech/)  
- **LLM Provider:** [OpenAI API](https://platform.openai.com/)  
- **Document Parsing:** PyPDF2 / python-docx / text loaders  
- **Embeddings:** OpenAI Embeddings API  

