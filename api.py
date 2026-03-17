from fastapi import FastAPI
from pydantic import BaseModel
from rag import process_urls, generate_answer

app = FastAPI(title="Real Estate RAG API")


class URLRequest(BaseModel):
    urls: list[str]


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def health_check():
    return {"status": "API is running"}


@app.post("/process-urls")
def process_urls_endpoint(request: URLRequest):

    steps = list(process_urls(request.urls))

    return {
        "message": "Vector DB created successfully",
        "steps": steps
    }


@app.post("/ask")
def ask_question(request: QueryRequest):

    answer, sources = generate_answer(request.query)

    return {
        "answer": answer,
        "sources": sources
    }