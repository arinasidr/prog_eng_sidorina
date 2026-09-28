from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from transformers import pipeline

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

app = FastAPI()

classifier = pipeline(
    "zero-shot-classification", 
    model=MODEL_NAME
)

class PredictionRequest(BaseModel):
    text: str
    categories: list[str]

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str):
        value = value.strip()

        if not value:
            raise ValueError("Текст должен быть не пустым")    

        return value

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, value: list[str]):
        categories = [
            category.strip()
            for category in value
            if category.strip()
        ]

        if not categories:
            raise ValueError("Необходимо указать хотя бы одну категорию")    
    
        return categories
    
class CategoryPrediction(BaseModel):
    label: str
    score: float

class PredictionResponse(BaseModel):
    label: str
    score: float
    predictions: list[CategoryPrediction]


@app.get('/')
def root():
    return {"message": "API работает"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    result = classifier(
        request.text,
        request.categories,
        multi_label=False
    )

    predictions = [
        {
            "label": label,
            "score": float(score),
        }
        for label, score in zip(result["labels"], result["scores"])
    ]

    return {
        "label": result["labels"][0],
        "score": float(result["scores"][0]),
        "predictions": predictions,
    }
