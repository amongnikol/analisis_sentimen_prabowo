import re
import time
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd

# Inisialisasi stemmer dan stopwords
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

# Mention penting yang ingin disimpan
mention_whitelist = ['prabowo']

# Kamus normalisasi kata informal → formal
normalisasi_kata = {
    'gak': 'tidak',
    'ga': 'tidak',
    'yg': 'yang',
    'aja': '',
    'amp': '',
    'nya': '',
    'ya': '',
    'gitu': '',
    'loh': '',
    'dong': '',
    'sih': '',
    'deh': '',
    'nih': '',
    'kok': '',
}

# Cache untuk stemming per proses (tidak bisa dipakai antar process)
def stem_cached(word, cache={}):
    if word not in cache:
        cache[word] = stemmer.stem(word)
    return cache[word]

# Fungsi preprocessing utama
def preprocessing(text):
    try:
        text = re.sub(r'@(\w+)', lambda m: m.group() if m.group(1).lower() in mention_whitelist else '', text)
        text = re.sub(r'(http\S+|www\S+|#(\w+)|@(\w+)|<.*?>)', ' ', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = text.lower()
        tokens = text.split()  # ganti dari word_tokenize untuk percepatan
        tokens = [normalisasi_kata.get(word, word) for word in tokens]
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2 and word != '']
        stemmed = [stem_cached(w) for w in tokens]
        return ' '.join(stemmed)
    except Exception as e:
        return ''  # tangani error jika ada teks yang bermasalah

# Fungsi untuk menjalankan preprocessing dengan multiprocessing
def parallel_preprocessing(df, column='full_text'):
    print(f"[INFO] Memulai preprocessing paralel pada kolom: {column}")
    start = time.time()
    texts = df[column].fillna('').astype(str).tolist()

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(preprocessing, texts), total=len(texts)))

    end = time.time()
    print(f"[INFO] Selesai preprocessing. Waktu total: {end - start:.2f} detik.")
    return results

# Contoh penggunaan
if __name__ == "__main__":
    # Contoh: load data
    df = pd.read_csv("dataset/text/prabowo_clean_text.csv")  # ganti nama file sesuai dataset kamu
    df['text_processed'] = parallel_preprocessing(df, column='full_text')

    # Simpan hasil
    df.to_csv("data_processed.csv", index=False)
    print("[INFO] Data berhasil disimpan ke 'data_processed.csv'")
