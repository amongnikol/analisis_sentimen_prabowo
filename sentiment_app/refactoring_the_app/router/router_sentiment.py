# router/sentiment_router.py
from fastapi import APIRouter, HTTPException
from dto.dto_sentiment import SentimentInput, SentimentOutput
from service.service_sentiment import SentimentService

router = APIRouter()
service = SentimentService()

@router.post("/predict", response_model=SentimentOutput)
def predict(input: SentimentInput):
    try:
        sentiment = service.predict_sentiment(input.text)
        return {"text": input.text, "sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
