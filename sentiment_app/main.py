from fastapi import FastAPI
from routes import router
from pydantic import BaseModel
from model import predict_sentiment

app = FastAPI()

class TextInput(BaseModel):
    text: str

class SentimentOutput(BaseModel):
    text: str
    sentiment: str

@app.post("/predict", response_model=SentimentOutput)
def predict(input: TextInput):
    result = predict_sentiment(input.text)
    return {"text": input.text, "sentiment": result}

# app.include_router(router)
