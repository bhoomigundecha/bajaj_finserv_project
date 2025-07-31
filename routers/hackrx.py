from fastapi import APIRouter, HTTPException
from models.schemas import AnalyzeRequest, AnalyzeResponse
from services.file_handler import fetch_and_extract_text
from services.chunker import chunk_text
from services.vectorstore import create_temp_index, embed_chunks_store
from services.llm_client import process_single_query

router = APIRouter()

@router.post("/hackrx/run", response_model=AnalyzeResponse)
async def run_analysis(request: AnalyzeRequest):
    try:
        print(f" Doc URL: {request.documents}")
        print(f" Questions: {request.questions}")

        raw_text = fetch_and_extract_text(request.documents)
        print(f" Text length: {len(raw_text)}")

        chunks = chunk_text(raw_text)
        print(f" Chunks generated: {len(chunks)}")

        index_name = create_temp_index()
        embed_chunks_store(index_name, chunks)

        answers = []
        for q in request.questions:
            ans = await process_single_query(q, index_name=index_name)
            answers.append(ans)

        return AnalyzeResponse(answers=answers)

    except Exception as e:
        print(f" ERROR: {repr(e)}")
        raise HTTPException(status_code=500, detail=str(e))
