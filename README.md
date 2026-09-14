# Yakut (Sakha) NER API

A REST API for Named Entity Recognition (NER) in the Yakut (Sakha) language,
built with FastAPI and a fine-tuned mBERT model hosted on HuggingFace.

## Overview

This service exposes a single endpoint that accepts raw Yakut-language text
and returns tagged entity spans (e.g. persons, locations, organizations).
Inference logic (`main.py`) is kept separate from model loading (`model.py`)
so the underlying checkpoint can be swapped or upgraded without touching the
API layer.

## Architecture

- **main.py** — FastAPI app, request/response schemas, route handling and
  input validation.
- **model.py** — Loads the tokenizer and token-classification model from
  HuggingFace via `transformers.pipeline`, and exposes a `predict()` method
  that returns structured entity dictionaries.
- Model weights are pulled dynamically from the HuggingFace Hub at startup,
  not bundled in the repo, keeping the codebase lightweight.

## Setup

```bash
git clone https://github.com/enifimov81k/yakut-ner-api.git
cd yakut-ner-api
python -m venv venv
venv\Scripts\Activate.ps1   # on Windows
# source venv/bin/activate    # on macOS/Linux
pip install -r requirements.txt
uvicorn main:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive
Swagger UI.

## API

### `GET /health`
Returns service status and whether the model has loaded successfully.

### `POST /predict`
Request body:
```json
{
  "text": "Саха сирэ түлэ"
}
```
Response body:
```json
{
  "entities": [
    {
      "text": "Саха",
      "label": "LOC",
      "start": 0,
      "end": 4,
      "score": 0.97
    }
  ]
}
```

## Model

Fine-tuned mBERT model for Yakut NER, hosted on HuggingFace:
`enifimov81k/yakut-ner-mbert` (replace with your actual model repo once
trained and uploaded).

## Notes on data and training

This repo contains the serving layer only. Fine-tuning requires a labeled
Yakut NER dataset in BIO/IOB2 format and a separate training script (not
included here). If no fine-tuned checkpoint exists yet, set the
`NER_MODEL_NAME` environment variable to a placeholder multilingual NER
model for local testing before your own checkpoint is ready.

## Version control

Managed with Git and GitHub. See `.gitignore` for excluded local artifacts.
