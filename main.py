from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load model
model_path = 'static/trained_model2.h5'
model = load_model(model_path, compile=False)

# Folder to temporarily store uploaded images
UPLOAD_FOLDER = 'static/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Fungsi untuk memprediksi gambar baru
def predict_image(image_path, model):
    img = image.load_img(image_path, target_size=(100, 100)) # Ubah ukuran gambar sesuai kebutuhan model
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0 # Normalisasi gambar sesuai praproses yang digunakan pada pelatihan model

    prediction = model.predict(img)
    return prediction
image_path = None

@app.route('/', methods=['GET', 'POST'])
def index():
    persentase_rusak=None
    persentase_tidak_rusak=None
    prediction = None
    global image_path
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_path = os.path.join(UPLOAD_FOLDER, 'temp.jpg') # Simpan file sementara
            uploaded_file.save(image_path)
            prediction_result = predict_image(image_path, model)
            rusak_prob = prediction_result[0][0]
            persentase_rusak = rusak_prob * 100
            persentase_tidak_rusak = 100 - persentase_rusak
            if persentase_rusak > 50:
                prediction = "Rusak"
            else:
                prediction = "Tidak Rusak"
            
    return render_template('index.html', prediction=prediction, persentase_rusak=persentase_rusak, persentase_tidak_rusak=persentase_tidak_rusak, image_path=image_path)

@app.route('/get_image/<path>')
def get_image(path):
    image_path = os.path.join(UPLOAD_FOLDER, path)  # Menggunakan path yang benar
    try:
        return send_from_directory(UPLOAD_FOLDER, path, mimetype='image/jpeg')
    finally:
        if filename == 'temp.jpg':
            os.remove(image_path)  # Hapus hanya jika filename adalah 'temp.jpg'

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)
