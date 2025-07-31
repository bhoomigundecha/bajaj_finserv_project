from fastapi import APIRouter, HTTPException
from models.schemas import AnalyzeRequest, AnalyzeResponse
from services.file_handler import fetch_and_extract_text
from services.chunker import chunk_text
from services.vectorstore import embed_chunks_store
from services.llm_client import process_multiquery_async  # Change this import

router = APIRouter()

@router.post("/hackrx/run", response_model=AnalyzeResponse)
async def run_analysis(request: AnalyzeRequest):
    try:
        # 1. Download + extract text
        raw_text = fetch_and_extract_text(request.documents)

        # 2. Chunk
        chunks = chunk_text(raw_text)

        # 3. Embed and store in Pinecone
        embed_chunks_store(chunks)

        # 4. Answer all queries (use async version)
        answers = await process_multiquery_async(request.questions)  # Add await

        return AnalyzeResponse(answers=answers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))