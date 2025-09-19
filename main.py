from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import PIL

def algila(image, model_path, labels_path):
    np.set_printoptions(suppress=True)
    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = image.convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

def ne(class_name):
    if "AI" in class_name:  
        return "Bu fotoğraf AI."
    elif "Gerçek" in class_name:
        return "Bu fotoğraf gerçek."
    else:
        return "Bu fotoğraf algılanamadı."

# Kendi fotoğraf dosya adını buraya yaz
path = 'images/resim.jpg'  # <-- Bu satırı değiştirmeyi unutma

try:
    image = PIL.Image.open(path)
    class_name, confidence_score = algila(image, "keras_model.h5", "labels.txt")
    oneri = ne(class_name)
    print(f"Tespit edilen foto: {class_name.title()} (%{confidence_score * 100:.2f} güven)\n{oneri}")

except FileNotFoundError:
    print(f"Hata: '{path}' adında bir dosya bulunamadı. Lütfen dosya adını ve yolunu kontrol edin.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")
