import os

class Settings:
    MODEL_PATH = os.getenv("MODEL_PATH", "best_model_GRU.h5")
    TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "tokenizer.pkl")
    MAX_LEN = int(os.getenv("MAX_LEN", 36))
