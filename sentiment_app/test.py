from model import predict_sentiment

# Contoh input
teks_uji_1 = "Saya sangat senang dengan program Prabowo"
teks_uji_2 = "Kinerja buruk dan mengecewakan"
teks_uji_3 = "Biasa saja, tidak terlalu bagus dan tidak terlalu buruk"

# Uji model
print("Input 1:", teks_uji_1)
print("Prediksi 1:", predict_sentiment(teks_uji_1))

print("Input 2:", teks_uji_2)
print("Prediksi 2:", predict_sentiment(teks_uji_2))

print("Input 3:", teks_uji_3)
print("Prediksi 3:", predict_sentiment(teks_uji_3))
