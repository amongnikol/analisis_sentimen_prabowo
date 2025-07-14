from repository.repository_sentiment import SentimentRepository
from model.model_sentiment import SentimentModel

class SentimentService:
    def __init__(self):
        self.repository = SentimentRepository()
        self.model = SentimentModel()

    def predict_sentiment(self, text: str) -> str:
        """Process the text and predict sentiment."""
        preprocessed_text = self.repository.preprocess_text(text)
        prediction = self.model.predict(preprocessed_text)
        return self.repository.get_sentiment_label(prediction)
