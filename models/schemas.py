from pydantic import BaseModel, HttpUrl
from typing import List

class AnalyzeRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class AnalyzeResponse(BaseModel):
    answers: List[str]
