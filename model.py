import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

MODEL_NAME = os.getenv("NER_MODEL_NAME", "enifimov81k/yakut-ner-mbert")


class NERModel:
    """
    Wraps a HuggingFace token-classification pipeline so the API layer
    (main.py) never touches model internals directly. Swapping model
    architectures or checkpoints only requires changing MODEL_NAME or
    this file.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._pipeline = None
        self._load()

    def _load(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self._pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )

    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def predict(self, text: str):
        raw = self._pipeline(text)
        entities = []
        for ent in raw:
            entities.append({
                "text": ent["word"],
                "label": ent["entity_group"],
                "start": int(ent["start"]),
                "end": int(ent["end"]),
                "score": float(ent["score"]),
            })
        return entities
