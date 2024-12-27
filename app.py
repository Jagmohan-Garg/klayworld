import os
import numpy as np
import pytesseract
import cv2
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from colorthief import ColorThief
from uuid import uuid4
from sklearn.cluster import KMeans
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
import pickle
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


app = Flask(__name__, static_folder="static")
CORS(app, origins=["*"])


pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
DATASET_DIR = r"static/dataset_images"
IMAGE_SIZE = (224, 224)
TOP_K = 6
NUM_CLUSTERS = 10


base_model = ResNet50(weights="imagenet", include_top=False, pooling="max")
vgg_model = VGG16(weights="imagenet", include_top=False, pooling="max")


yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def safe_normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return np.zeros_like(vector)
    return vector / norm

def extract_multiscale_features(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.LANCZOS)
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
        return base_model.predict(image_array).flatten()
    except Exception as e:
        print(f"Error extracting multiscale features: {e}")
        return None

def extract_texture_features(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.LANCZOS)
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
        return vgg_model.predict(image_array).flatten()
    except Exception as e:
        print(f"Error extracting texture features: {e}")
        return None

def extract_text_from_image(image_path):
    try:
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray_image)
        return text.strip()
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def detect_objects(image_path):
    try:
        results = yolo_model(image_path)
        detections = results.pandas().xyxy[0]
        return [
            {"label": row["name"], "confidence": row["confidence"], "bbox": row[["xmin", "ymin", "xmax", "ymax"]].tolist()}
            for _, row in detections.iterrows()
        ]
    except Exception as e:
        print(f"Error detecting objects: {e}")
        return []

def extract_dominant_colors(image_path, n_colors=5):
    try:
        image = Image.open(image_path).convert("RGB")
        image.save("temp_image.jpg")
        color_thief = ColorThief("temp_image.jpg")
        palette = color_thief.get_palette(color_count=n_colors)
        os.remove("temp_image.jpg")
        return palette if palette else [(0, 0, 0)] * n_colors
    except Exception as e:
        print(f"Error extracting dominant colors: {e}")
        return [(0, 0, 0)] * n_colors

def rgb_to_lab(rgb_color):
    """Convert RGB to LAB color space."""
    srgb = sRGBColor(rgb_color[0] / 255.0, rgb_color[1] / 255.0, rgb_color[2] / 255.0)
    return convert_color(srgb, LabColor)

def color_similarity_ciede2000(rgb1, rgb2):
    """Compute color similarity using CIEDE2000."""
    try:
        lab1 = rgb_to_lab(rgb1)
        lab2 = rgb_to_lab(rgb2)
        delta_e = delta_e_cie2000(lab1, lab2)
        return 1 / (1 + delta_e) 
    except Exception as e:
        print(f"Error in CIEDE2000 computation: {e}")
        return 0

def compute_similarity(uploaded_features, entry, uploaded_texture, search_type, uploaded_palette=None):
    try:
       
        uploaded_features = safe_normalize(uploaded_features)
        entry_features = safe_normalize(entry["multiscale"])
        
        uploaded_texture = safe_normalize(uploaded_texture)
        entry_texture = safe_normalize(entry["texture"])
        
        
        cosine_sim = np.dot(uploaded_features, entry_features)
        
        
        texture_sim = np.dot(uploaded_texture, entry_texture)
        
        
        if uploaded_palette and entry["palette"]:
            color_sim = color_similarity_ciede2000(uploaded_palette[0], entry["palette"][0])
        else:
            color_sim = 0  

        if search_type == "texture":
            return 0.7 * cosine_sim + 0.3 * texture_sim 
        elif search_type == "color":
            return 0.9 * color_sim + 0.1 * texture_sim
        else:
            return 0.2 * cosine_sim + 0.2 * texture_sim  + 0.6 * color_sim 
    except Exception as e:
        print(f"Error in similarity computation: {e}")
        return 0

def precompute_dataset_features(dataset_path, n_clusters, cache_file="precomputed_features.pkl"):
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    features_list, filenames = [], []
    for filename in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, filename)
        if os.path.isfile(image_path) and image_path.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                multiscale_features = extract_multiscale_features(image_path)
                texture_features = extract_texture_features(image_path)
                palette = extract_dominant_colors(image_path)
                if multiscale_features is not None and texture_features is not None:
                    features_list.append({
                        "multiscale": multiscale_features,
                        "texture": texture_features,
                        "palette": palette
                    })
                    filenames.append(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    feature_vectors = [f["multiscale"] for f in features_list]
    kmeans = KMeans(n_clusters=min(n_clusters, len(feature_vectors)), random_state=42)
    cluster_labels = kmeans.fit_predict(feature_vectors) if len(feature_vectors) > 1 else [0] * len(feature_vectors)
    with open(cache_file, "wb") as f:
        pickle.dump((features_list, filenames, cluster_labels, kmeans), f)
    return features_list, filenames, cluster_labels, kmeans


precomputed_features, filenames, cluster_labels, kmeans_model = precompute_dataset_features(DATASET_DIR, NUM_CLUSTERS)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400

    file_path = f"./temp_{uuid4().hex}_{file.filename}"
    file.save(file_path)

    search_type = request.form.get("search_type", "default").lower()

    try:
        uploaded_features = extract_multiscale_features(file_path)
        uploaded_texture = extract_texture_features(file_path)
        uploaded_palette = extract_dominant_colors(file_path)
        text_detected = extract_text_from_image(file_path)
        objects_detected = detect_objects(file_path)

        if uploaded_features is None or uploaded_texture is None:
            return jsonify({"error": "Feature extraction failed."}), 500

        response = {
            "text_detected": text_detected,
            "objects_detected": objects_detected,
            "top_matches": [],
        }

        if precomputed_features:
            similarities = []
            for idx, entry in enumerate(precomputed_features):
                score = compute_similarity(
                    uploaded_features,
                    entry,
                    uploaded_texture,
                    search_type,
                    uploaded_palette=uploaded_palette
                )
                similarities.append({
                    "image": os.path.join("static", "dataset_images", filenames[idx]),
                    "score": round(float(score), 2),
                })
            similarities.sort(key=lambda x: x["score"], reverse=True)
            response["top_matches"] = similarities[:TOP_K]

        return jsonify(response)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
