import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

MODEL_PATH = "model/skin_model.keras"
IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)

img_path = "sample_images/test1.jpg"

img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
class_index = np.argmax(pred)
confidence = np.max(pred) * 100

classes = [
    "Atopic Dermatitis",
    "Basal Cell Carcinoma",
    "Benign Keratosis-like Lesions",
    "Eczema",
    "Melanocytic Nevi",
    "Melanoma"
]

print("Predicted Disease:", classes[class_index])
print("Confidence:", round(confidence, 2), "%")
