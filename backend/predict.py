from flask import Blueprint, render_template, request, flash
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import tensorflow as tf

predict_bp = Blueprint("predict", __name__)

# -------- Correct upload folder path --------
UPLOAD_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "static",
    "uploads"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "model/skin_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "Atopic Dermatitis",
    "Basal Cell Carcinoma",
    "Benign Keratosis-like Lesions",
    "Eczema",
    "Melanocytic Nevi",
    "Melanoma"
]


def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@predict_bp.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return render_template("index.html")

    if "image" not in request.files:
        flash("No file uploaded")
        return render_template("index.html")

    file = request.files["image"]

    if file.filename == "":
        flash("No file selected")
        return render_template("index.html")

    filename = secure_filename(file.filename)

    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    print("Saved file at:", save_path)

    print("Saved file:", save_path)
    print("Running model prediction...")

    # ===== REAL PREDICTION =====
    img = preprocess_image(save_path)

    preds = model.predict(img)
    pred_index = np.argmax(preds[0])
    confidence = float(preds[0][pred_index])
    predicted_class = CLASS_NAMES[pred_index]

    print("Raw predictions:", preds)
    print("Predicted index:", pred_index)
    print("Confidence:", confidence)
    print("Class:", predicted_class)

    return render_template(
    "result.html",
    prediction=predicted_class,
    confidence=round(confidence * 100, 2),
    image_url="/static/uploads/" + filename
    )
