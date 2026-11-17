"""FastAPI-бэкенд: предсказание quality белого вина."""
import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Wine Quality Predictor",
    description="Регрессия: предсказание экспертной оценки quality белого вина (шкала ~3–9).",
)

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.joblib"))
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.is_file():
            return None
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class WineInput(BaseModel):
    fixed_acidity: float = Field(..., ge=0, description="Фиксированная кислотность, г/дм³")
    volatile_acidity: float = Field(..., ge=0, description="Летучая кислотность, г/дм³")
    citric_acid: float = Field(..., ge=0, description="Лимонная кислота, г/дм³")
    residual_sugar: float = Field(..., ge=0, description="Остаточный сахар, г/дм³")
    chlorides: float = Field(..., ge=0, description="Хлориды, г/дм³")
    free_sulfur_dioxide: float = Field(..., ge=0, description="Свободный SO₂, мг/дм³")
    total_sulfur_dioxide: float = Field(..., ge=0, description="Общий SO₂, мг/дм³")
    density: float = Field(..., gt=0, description="Плотность, г/см³")
    pH: float = Field(..., ge=0, le=14, description="pH")
    sulphates: float = Field(..., ge=0, description="Сульфаты, г/дм³")
    alcohol: float = Field(..., ge=0, description="Алкоголь, % об.")


# Порядок должен совпадать с FEATURES при обучении
FEATURE_ORDER = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


@app.get("/health")
def health():
    return {"model_ready": MODEL_PATH.is_file()}


@app.post("/predict")
def predict(sample: WineInput):
    bundle = get_bundle()
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model missing; run training first (python -m src.train)",
        )

    model = bundle["model"]
    values = [[getattr(sample, name) for name in FEATURE_ORDER]]
    pred = float(model.predict(values)[0])

    return {
        "predicted_quality": round(pred, 2),
        "unit": "expert score (approx. 3–9)",
        "interpretation": _interpret(pred),
    }


def _interpret(score: float) -> str:
    if score < 5:
        return "Ниже среднего качества"
    if score < 6:
        return "Среднее качество"
    if score < 7:
        return "Хорошее качество"
    return "Высокое качество"
