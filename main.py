from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load model
model_path = 'static/trained_model2.h5'
model = load_model(model_path, compile=False)

# # Folder to temporarily store uploaded images
# UPLOAD_FOLDER = 'static'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# Fungsi untuk memprediksi gambar baru
def predict_image(image_path, model):
    img = image.load_img(image_path, target_size=(100, 100)) # Ubah ukuran gambar sesuai kebutuhan model
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0 # Normalisasi gambar sesuai praproses yang digunakan pada pelatihan model

    prediction = model.predict(img)
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    persentase_rusak=None
    persentase_tidak_rusak=None
    prediction = None
    image_path = None
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_path = 'static/temp.jpg' # Simpan file sementara
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
            # # Hapus file temp.jpg setelah selesai diproses
            # if os.path.exists(image_path):
            #     os.remove(image_path)
            
            # # Hapus file temp.jpg setelah selesai diproses
            # if os.path.exists(image_path):
            #     os.remove(image_path)

            # # Setelah selesai prediksi, hapus gambar
            # @after_this_request
            # def remove_temp(response):
            #     os.remove(image_path)
            #     return response
    
            # # Hapus gambar sementara setelah proses prediksi selesai
            # if os.path.exists(image_path):
            #     os.remove(image_path)
            #     image_path = None

            # # Hapus file sementara setelah selesai prediksi
            # os.remove(image_path)
            # image_path = None  # Set image_path ke None setelah dihapus


# @app.route('/get_image')
# def get_image():
#     image_path = request.args.get('path')
#     return image_path

# @app.route('/get_image')
# def get_image():
#     image_path = request.args.get('path')
#     # Lakukan validasi terhadap image_path jika diperlukan
#     return send_file(image_path, mimetype='image/jpeg')

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)
