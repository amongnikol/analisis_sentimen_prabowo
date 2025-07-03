import re
import pickle
import numpy as np
import string
import logging

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# SETUP LOGGER 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Model & Tokenizer
try:
    model = tf.keras.models.load_model("best_model_GRU.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
except Exception as e:
    logger.exception("Gagal memuat model atau tokenizer: %s", e)
    raise

# Preprocessing Setup
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))
mention_whitelist = ['prabowo']

normalisasi_kata = {
    'gak': 'tidak','ga': 'tidak','yg': 'yang','utk': 'untuk',
    'dgn': 'dengan','apik': 'baik','sdh': 'sudah','krn': 'karena',
    'apik': 'bagus','tdk': 'tidak','klo': 'kalo','sbg': 'sebagai',
    'gue': 'aku','dlm': 'dalam','jgn': 'jangan','jkw': 'jokowi','org': 'orang',
    'nggak': 'tidak','aja': '','amp': '','nya': '','ya': '','gitu': '','loh': '',
    'dong': '','sih': '','deh': '','nih': '','kok': '',
}

def preprocessing(text):
    text = re.sub(r'@(\w+)', lambda m: m.group() if m.group(1).lower() in mention_whitelist else '', text)
    text = re.sub(r'http\S+|www\S+|<.*?>|#', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    
    tokens = word_tokenize(text)
    tokens = [normalisasi_kata.get(word, word) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2 and word != '']
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

# ======================
# === Predict Function
# ======================
MAX_LEN = 36  # Sesuaikan dengan yang kamu pakai saat training

def predict_sentiment(text: str) -> str:
    try:
        if len(text.strip()) < 5:
            return "error: Teks terlalu pendek untuk dianalisis."
        
        cleaned = preprocessing(text)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
        
        prediction = model.predict(padded)[0]
        label = np.argmax(prediction)

        if label == 0:
            return "netral"
        elif label == 1:
            return "positif"
        elif label == 2:
            return "negatif"
        else:
            return "error: Label tidak dikenali"
        
    except Exception as e:
        logger.exception("Terjadi kesalahan saat prediksi sentimen: %s", e)
        return "error: Terjadi kesalahan internal"
