from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import process_blob_and_answer
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

class QueryRequest(BaseModel):
    blob_url: str
    question: str

@app.post("/query-pdf")
def query_pdf(request: QueryRequest):
    try:
        answer = process_blob_and_answer(request.blob_url, request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
