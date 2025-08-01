from fastapi import FastAPI
from routers import hackrx

app = FastAPI(
    title="Document Analyzer API",
    version="1.0"
)

app.include_router(hackrx.router)
