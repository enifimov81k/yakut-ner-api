from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from model import NERModel

app = FastAPI(
    title="Yakut (Sakha) NER API",
    description="REST API for Named Entity Recognition in the Yakut (Sakha) language, "
                 "backed by a fine-tuned mBERT model.",
    version="1.0.0",
)

ner_model = NERModel()


class TextRequest(BaseModel):
    text: str


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    score: float


class NERResponse(BaseModel):
    entities: List[Entity]


@app.get("/")
def root():
    return {"status": "ok", "message": "Yakut NER API is running. See /docs for usage."}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": ner_model.is_loaded()}


@app.post("/predict", response_model=NERResponse)
def predict(request: TextRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' must not be empty.")
    try:
        entities = ner_model.predict(request.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
    return NERResponse(entities=entities)
