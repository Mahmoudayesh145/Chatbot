from pydantic import BaseModel
from typing import Optional, List, Any

class GenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 200

class SummarizationRequest(BaseModel):
    text: str
    max_length: Optional[int] = 130
    min_length: Optional[int] = 30

class TranslationRequest(BaseModel):
    text: str
    source_lang: Optional[str] = "en"
    target_lang: Optional[str] = "ar"

class EmotionRequest(BaseModel):
    text: str

class SentimentRequest(BaseModel):
    text: str

class GeminiRequest(BaseModel):
    prompt: str
    history: Optional[List[Any]] = []
