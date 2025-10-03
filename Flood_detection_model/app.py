import os
import torch
import torch.nn.functional as F
import numpy as np
from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForSemanticSegmentation,AutoConfig
from PIL import Image
import tifffile as tiff
from flask import send_file
import io

# ----------------
# CONFIG
# ----------------
MODEL_PATH = "C:\\Users\\gamal\\PycharmProjects\\Flood_detection_model\\segformer_model"  # path to your saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------
# LOAD MODEL
# ----------------
config = AutoConfig.from_pretrained(MODEL_PATH, num_labels=2)  # set your correct number of classes
model = AutoModelForSemanticSegmentation.from_pretrained(
    MODEL_PATH,
    config=config,
    ignore_mismatched_sizes=True  # allows loading your trained head
)
model.to(device)
model.eval()

# ----------------
# FLASK APP
# ----------------
app = Flask(__name__)

def load_image(path_or_file):
    if isinstance(path_or_file, str) and path_or_file.endswith(".tif"):
        img = tiff.imread(path_or_file).astype(np.float32)
        img = img[:, :, [3, 2, 1]]  # picking bands
    else:
        img = np.array(Image.open(path_or_file).convert("RGB")).astype(np.float32)  # force RGB

    img = img / 255.0  # normalize to [0,1]
    return img

def predict_segmentation(image):
    img_tensor = torch.tensor(image).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(pixel_values=img_tensor).logits
        pred = F.interpolate(pred, size=(image.shape[0], image.shape[1]),
                             mode="bilinear", align_corners=False)
        pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

    return pred_mask

@app.route("/", methods=["GET"])
def home():
    return "Segmentation Flask API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = load_image(file)

    X = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(pixel_values=X).logits
        pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    # convert mask → image
    mask_img = Image.fromarray(pred_mask * 255)  # scale classes to grayscale

    # save to memory buffer
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)

    return send_file(buf, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
