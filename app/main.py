from fastapi import FastAPI
from pydantic import BaseModel

from app.pipeline.process_pipeline import process_pipeline

app = FastAPI(title="MT-Sense AI-Service")


class AnalyzeRequest(BaseModel):
    texts: list[str]
    confidence_threshold: float = 0.55


class AnalyzeResultItem(BaseModel):
    text: str
    cleaned_text: str
    sentiment_label: str
    sentiment_score: float
    confidence: float
    categories: list[str]
    reason: str
    low_confidence: bool


class AnalyzeResponse(BaseModel):
    results: list[AnalyzeResultItem]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    results = process_pipeline(
        raw_texts=request.texts,
        confidence_threshold=request.confidence_threshold,
    )
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
