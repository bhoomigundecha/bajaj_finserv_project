from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    documents: str
    questions: list[str]

class AnalyzeResponse(BaseModel):
    answers: list[str]

