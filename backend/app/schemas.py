from pydantic import BaseModel
from typing import List


class PredictRequest(BaseModel):
    image_base64: str


class PredictResponse(BaseModel):
    label: str
    fake_confidence: float
    real_confidence: float


class BatchPredictRequest(BaseModel):
    image_base64_list: List[str]


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
