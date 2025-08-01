from fastapi import APIRouter, HTTPException
from models.schemas import AnalyzeRequest, AnalyzeResponse
from services.file_handler import fetch_and_extract_text
from services.chunker import chunk_text
from services.vectorstore import ensure_collection, embed_chunks_store
from services.llm_client import process_single_query

router = APIRouter()

@router.post("/hackrx/run", response_model=AnalyzeResponse)
async def run_analysis(request: AnalyzeRequest):
    try:
        raw_text = fetch_and_extract_text(request.documents)
        chunks = chunk_text(raw_text)

        doc_id = "doc"  
        ensure_collection()
        embed_chunks_store(doc_id, chunks)

        answers = []
        for q in request.questions:
            ans = await process_single_query(q, doc_id=doc_id)
            answers.append(ans)

        return AnalyzeResponse(answers=answers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
