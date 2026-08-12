from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    question: str
    document:List[str]
    answer:str

class PredictResponse(BaseModel):
    label: str
    confidence: float