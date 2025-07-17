# repository/sentiment_repository.py
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from enums.enum_sentiment import SentimentEnum
import numpy as np
import re

class SentimentRepository:
    def __init__(self):
        self.stemmer = StemmerFactory().create_stemmer()
        self.stop_words = set(stopwords.words('indonesian'))
        self.mention_whitelist = {'prabowo'}
        self.normalisasi_kata = {
            'gak': 'tidak', 'ga': 'tidak', 'yg': 'yang', 'utk': 'untuk', 'dgn': 'dengan',
            'apik': 'baik', 'sdh': 'sudah', 'krn': 'karena', 'apik': 'bagus', 'tdk': 'tidak'
        }

    def clean_text(self, text: str) -> str:
        """Clean text by removing unwanted characters."""
        text = re.sub(r'@(\w+)', lambda m: m.group() if m.group(1).lower() in self.mention_whitelist else '', text)
        text = re.sub(r'http\S+|www\S+|<.*?>|#', ' ', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        return text.lower()
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess the text (tokenize, normalize, remove stop words, and stem)."""
        text = self.clean_text(text)
        tokens = word_tokenize(text)
        tokens = [self.normalisasi_kata.get(word, word) for word in tokens]
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        tokens = [self.stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)

    def get_sentiment_label(self, prediction: list) -> str:
        """Get sentiment label based on prediction."""
        label = int(np.argmax(prediction)) # pastikan outputnya int
        return SentimentEnum.from_index(label).value
