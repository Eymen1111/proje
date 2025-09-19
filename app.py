import os
import cv2
import numpy as np
import torch
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from PIL import Image
from rembg import remove  # rembg kütüphanesini dahil et

try:
    from main import algila, ne
except ImportError:
    print(" 'algila' ve 'ne' fonksiyonları bulunamadı. ")
    algila = None
    ne = None


app = Flask(__name__)


UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed_videos"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def change_back(background, img):
    x, y = 0, 0

    
    background = cv2.resize(background, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    res = np.copy(background)
    place = res[y:y + img.shape[0], x:x + img.shape[1]]

    # Resmin alfa kanalı varsa
    if img.shape[2] == 4:
        a = img[..., 3:].repeat(3, axis=2).astype('uint16')
        place[...] = (place.astype('uint16') * (255 - a) // 255) + img[..., :3].astype('uint16') * a // 255
    else:
        
        place[..., :3] = img

    return res




def process_video(video_path):
    try:
        model = YOLO("yolov8n.pt")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Hata: {video_path} dosyası açılamadı.")
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"processed_{base_name}.webm" 
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)

        
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, save=False)
            im_array = results[0].plot()
            out.write(im_array)

        cap.release()
        out.release()
        return output_filename

    except Exception as e:
        print(f"Video işlenirken bir hata oluştu: {e}")
        return None



@app.route('/')
def home():
    """Ana sayfa - Giriş ekranını gösterir."""
    return render_template('giris.html')


@app.route('/sayfa1')
def sayfa1():
    """İlk arayüz sayfasını (resim yükleme) gösterir."""
    return render_template('index.html')


@app.route('/sayfa2')
def sayfa2():
    """İkinci arayüz sayfasını (video yükleme) gösterir."""
    return render_template('index2.html')


@app.route('/sayfa3')
def sayfa3():
    return render_template('index3.html')


@app.route('/change_background', methods=['GET', 'POST'])
def change_background_route():
    if request.method == 'GET':
        return render_template('index3.html')

    if request.method == 'POST':
        if 'image_file' not in request.files or 'background_file' not in request.files:
            return "Lütfen hem ön plan hem de arka plan resmini seçin."

        image_file = request.files['image_file']
        background_file = request.files['background_file']

        if image_file.filename == '' or background_file.filename == '':
            return "Dosya seçilmedi."

        
        image_filename = secure_filename(image_file.filename)
        background_filename = secure_filename(background_file.filename)

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        background_path = os.path.join(app.config['UPLOAD_FOLDER'], background_filename)

        image_file.save(image_path)
        background_file.save(background_path)

        try:
            
            original_image_bytes = open(image_path, "rb").read()
            processed_image_bytes = remove(original_image_bytes) 

            
            processed_image_filename = "processed_" + image_filename
            processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_image_filename)

            with open(processed_image_path, "wb") as out_file:
                out_file.write(processed_image_bytes)

            
            image = cv2.imread(processed_image_path, cv2.IMREAD_UNCHANGED)
            background = cv2.imread(background_path, cv2.IMREAD_UNCHANGED)

            if image is None or background is None:
                return "Resimlerden biri okunamadı. Lütfen geçerli bir resim dosyası yükleyin."

            result_image = change_back(background, image)

            
            result_filename = "result_" + image_filename
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            cv2.imwrite(result_path, result_image)

            return render_template('sonuc3.html', result_filename=result_filename)

        except Exception as e:
            return f"Resim işlenirken bir hata oluştu: {e}"


@app.route('/upload', methods=['POST'])
def upload_file():
    
    if algila is None or ne is None:
        return "Resim işleme fonksiyonları bulunamadı. Lütfen 'main.py' dosyasını kontrol edin."

    if 'file' not in request.files or request.files['file'].filename == '':
        return "Dosya seçilmedi veya yüklenmedi."

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    image = Image.open(filepath)
    class_name, confidence_score = algila(image, "keras_model.h5", "labels.txt")
    oneri = ne(class_name)

    return render_template(
        'sonuc.html',
        class_name=class_name,
        confidence=confidence_score,
        result_text=oneri
    )


@app.route('/result/<filename>')
def result(filename):
    return render_template('sonuc2.html', video_filename=filename)


@app.route('/processed_videos/<filename>')
def serve_processed_video(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/process_video', methods=['POST'])
def process_video_route():
    if 'file' not in request.files or request.files['file'].filename == '':
        return 'Dosya seçilmedi veya yüklenmedi.'

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    processed_filename = process_video(filepath)
    if processed_filename:
        return redirect(url_for('result', filename=processed_filename))
    else:
        return 'Video işlenirken bir hata oluştu.'


if __name__ == '__main__':
    app.run(debug=True)
