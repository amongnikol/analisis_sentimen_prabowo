from pydantic import BaseModel

class SentimentInput(BaseModel):
    text: str

class SentimentOutput(BaseModel):
    text: str
    sentiment: str