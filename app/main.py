import asyncio
import hmac
import logging
from io import BytesIO

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.config import AI_TRAINING_TOKEN
from app.knowledge_base.compiler import KnowledgeCompileError, compile_knowledge_base
from app.knowledge_base.qa import KnowledgeQAError, answer_question
from app.nlp.keywords import extract_keywords
from app.ml.predict import activate_model
from app.pipeline.process_pipeline import process_pipeline
from scripts.train_model import load_and_clean_data, save_model, train_with_metrics

app = FastAPI(title="MT-Sense AI-Service")
training_lock = asyncio.Lock()
logger = logging.getLogger(__name__)


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


class KeywordsRequest(BaseModel):
    texts: list[str]


class KeywordsResponse(BaseModel):
    keywords: list[list[str]]


class TrainingResponse(BaseModel):
    accuracy: float
    macroF1: float
    testRows: int
    trainingRows: int


class KnowledgeCompileRequest(BaseModel):
    source: dict
    previous_articles: list[dict] = Field(default_factory=list)


class KnowledgeCompileResponse(BaseModel):
    title_th: str
    title_en: str
    summary_th: str
    summary_en: str
    markdown: str
    tags: list[str]
    related_period_ids: list[str]
    suggested_questions: list[str]


class KnowledgeQARequest(BaseModel):
    question: str
    articles: list[dict] = Field(default_factory=list)
    locale: str = "th"


class KnowledgeQAResponse(BaseModel):
    answer: str
    used_period_ids: list[str]


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


@app.post("/keywords", response_model=KeywordsResponse)
def keywords(request: KeywordsRequest):
    if len(request.texts) > 200:
        raise HTTPException(status_code=413, detail="at most 200 comments per request")
    return {"keywords": [extract_keywords(text) for text in request.texts]}


@app.post("/knowledge/compile", response_model=KnowledgeCompileResponse)
def compile_knowledge(request: KnowledgeCompileRequest):
    try:
        return compile_knowledge_base(request.source, request.previous_articles)
    except KnowledgeCompileError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/knowledge/ask", response_model=KnowledgeQAResponse)
def ask_knowledge(request: KnowledgeQARequest):
    try:
        return answer_question(request.question, request.articles, request.locale)
    except KnowledgeQAError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _train_uploaded_workbook(data: bytes) -> dict:
    dataframe = load_and_clean_data(BytesIO(data))
    model, vectorizer, report = train_with_metrics(dataframe)
    save_model(model, vectorizer)
    activate_model(model, vectorizer)
    return report


@app.post("/train", response_model=TrainingResponse)
async def train_model_endpoint(request: Request, x_training_token: str | None = Header(default=None)):
    if not AI_TRAINING_TOKEN:
        raise HTTPException(status_code=503, detail="AI training token is not configured")
    if not x_training_token or not hmac.compare_digest(x_training_token, AI_TRAINING_TOKEN):
        raise HTTPException(status_code=403, detail="not authorized to train")
    if training_lock.locked():
        raise HTTPException(status_code=409, detail="model training is already running")

    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > (10 << 20):
            raise HTTPException(status_code=413, detail="workbook exceeds 10 MB")
    if len(data) < 4 or bytes(data[:2]) != b"PK":
        raise HTTPException(status_code=400, detail="upload an .xlsx workbook")

    async with training_lock:
        try:
            return await run_in_threadpool(_train_uploaded_workbook, bytes(data))
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid training workbook: {exc}") from exc
        except Exception as exc:
            logger.exception("model training failed")
            raise HTTPException(status_code=500, detail="model training failed") from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
