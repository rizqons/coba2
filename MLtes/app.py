import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
from flask import Flask, request, jsonify

import os

app = Flask(__name__)

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path='model_vericheck.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

df = pd.read_csv('merging_data.csv')
features = df['headline'] + df['content']
labels = df['hoax']

training_sentences, validation_sentences, training_labels, validation_labels = train_test_split(features, labels, train_size=.8, random_state=42)

vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

input_text = """
= = =

Narasi:

“Salah Satu Mesjid di Palestina Dibom Saat Lagi Adzan”

= = =

Penjelasan:

Beredar sebuah video di Facebook yang menunjukkan sebuah bangunan yang diklaim merupakan masjid di Palestina yang dibom saat mengumandangkan azan.

Setelah ditelusuri klaim tersebut menyesatkan. Faktanya video asli telah beredar di YouTube pada 2014, dari judul video tersebut menyebutkan bahwa bangunan tersebut adalah Kuil Uwais al-Qarni yang merupakan bangunan dari makam dari Uwais al-Qarni yang dihancurkan oleh ISIS. Dilansir dari LiputanIslam.com, ISIS telah menghancurkan tempat tersebut sebagai bentuk penegakan tauhid dan menjauhi segala bentuk kesyirikan.

Dengan demikian, video masjid dibom saat mengumandangkan azan di Palestina adalah tidak benar dengan kategori Konten yang Menyesatkan.

= = =
"""

# GET
@app.route("/")
def predict_text_tflite():
    text = input_text
    # Preprocessing text before going to input
    text = text.replace('=', '')
    text = re.sub(r'\[.?\]|\(.?\)', '', text)
    text = re.sub(r'\s+', ' ', text)

    # Get the input to input tensor
    input_texts = [text]
    # Tokenizing Text with TensorFlow Tokenizer
    input_seq = tokenizer.texts_to_sequences(input_texts)
    input_padded = pad_sequences(input_seq, maxlen=max_length, truncating=trunc_type, padding=padding_type)
    input_data = np.array(input_padded, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # Get the prediciton value from tennsor output
    output_data = interpreter.get_tensor(output_details[0]['index'])

    prediction = output_data[0][0]
    # return print("Predicition Value:", prediction)
    return f"Prediction Value: {prediction}"


# POST
@app.route("/predict", methods=["GET", "POST"])
def predict_text_tflite_post():
    if request.method == "POST":
        data = request.get_json()  # Mengasumsikan data input dikirim sebagai JSON
        text = data.get("text", "")  # Sesuaikan kunci berdasarkan struktur JSON Anda

        # Pra-pemrosesan teks sebelum dimasukkan ke dalam input
        text = text.replace('=', '')
        text = re.sub(r'\[.?\]|\(.?\)', '', text)
        text = re.sub(r'\s+', ' ', text)

        # Mendapatkan input untuk tensor input
        input_texts = [text]
        # Tokenisasi Teks dengan TensorFlow Tokenizer
        input_seq = tokenizer.texts_to_sequences(input_texts)
        input_padded = pad_sequences(input_seq, maxlen=max_length, truncating=trunc_type, padding=padding_type)
        input_data = np.array(input_padded, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        # Mendapatkan nilai prediksi dari tensor output
        output_data = interpreter.get_tensor(output_details[0]['index'])

        prediction = output_data[0][0]

        # Mengembalikan prediksi sebagai JSON
        return jsonify({"prediksi": float(prediction)})

# prediction = predict_text_tflite(input_text)
# print("Predicition Value:", prediction)

if __name__ == "__main__":
    # Used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an entrypoint to app.yaml.
     app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))