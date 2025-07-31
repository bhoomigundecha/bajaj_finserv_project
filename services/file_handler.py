import requests
import pdfplumber
import docx
import tempfile
import os
from fastapi import HTTPException

def fetch_and_extract_text(file_url: str) -> str:
    try:
        response = requests.get(file_url)
        if response.status_code != 200:
            raise Exception("download issue")

        content_type = response.headers.get("content-type", "").lower()
        _, temp_path = tempfile.mkstemp()

        with open(temp_path, "wb") as f:
            f.write(response.content)

        file_url_str = str(file_url).lower()

        if ".pdf" in file_url_str or "pdf" in content_type:
            text = extract_pdf(temp_path)
        elif ".docx" in file_url_str or "word" in content_type:
            text = extract_docx(temp_path)
        elif ".txt" in file_url_str or "text" in content_type:
            text = extract_txt(temp_path)
        else:
            raise Exception("file type match issue")

        os.remove(temp_path)
        return text

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"file processing issue: {e}")

def extract_pdf(path: str) -> str:
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        raise Exception(f"pdf: {e}")

def extract_docx(path: str) -> str:
    try:
        doc = docx.Document(path)
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        raise Exception(f"doc: {e}")

def extract_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise Exception(f"txt: {e}")
