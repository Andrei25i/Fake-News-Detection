from pydantic import BaseModel
from typing import Dict, List

class ModelPrediction(BaseModel):
    model_used: str
    prediction: str
    confidence_score: float
    confidence_percent: str
    probabilities: Dict[str, float]

class NewsRequest(BaseModel):
    title: str
    text: str

class NewsResponse(BaseModel):
    predictions: List[ModelPrediction]