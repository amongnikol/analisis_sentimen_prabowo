import tensorflow as tf
import pickle
from config.settings import Settings
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SentimentModel:
    def __init__(self):
        self.model = self.load_model(Settings.MODEL_PATH)
        self.tokenizer = self.load_tokenizer(Settings.TOKENIZER_PATH)
    
    def load_model(self, model_path: str):
        """Load the trained sentiment analysis model."""
        return tf.keras.models.load_model(model_path)
    
    def load_tokenizer(self, tokenizer_path: str):
        """Load the tokenizer."""
        with open(tokenizer_path, "rb") as f:
            return pickle.load(f)
    
    def predict(self, text: str):
        """Preprocess and predict the sentiment of the text."""
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=Settings.MAX_LEN, padding='post', truncating='post')
        return self.model.predict(padded)[0]
