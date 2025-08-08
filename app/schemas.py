from pydantic import BaseModel
from typing import List

class NewsClassifyRequest(BaseModel):
    text: str

class NewsClassifyResponse(BaseModel):
    label: str
    score: float

class OCRResponse(BaseModel):
    text: str

class NewsArticle(BaseModel):
    title: str
    content: str
    url: str

class NewsRecommendResponse(BaseModel):
    articles: List[NewsArticle]

