import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import v2
from PIL import Image
import numpy as np
import cv2
import os

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load Model
# -------------------------------
def load_model(model_path):
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[1] = nn.Linear(1280, 2)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model("models/best.pth")

# -------------------------------
# Labels
# -------------------------------
labels = ["male", "female"]

# -------------------------------
# Transform
# -------------------------------
transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32),
    v2.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------------------------------
# Load Haar Cascade (safe path)
# -------------------------------
cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

# -------------------------------
# Prediction Function
# -------------------------------
def predict_image(image, return_face=False):

    # If path → load image
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    img_np = np.array(image)

    # Resize for better detection
    resized = cv2.resize(img_np, (500, 500))
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(50, 50)
    )

    face_img = None

    # If face found → map back to original size
    if len(faces) > 0:
        x, y, w, h = faces[0]

        h_orig, w_orig, _ = img_np.shape

        x = int(x * w_orig / 500)
        y = int(y * h_orig / 500)
        w = int(w * w_orig / 500)
        h = int(h * h_orig / 500)

        # Clamp values
        x, y = max(0, x), max(0, y)
        x2 = min(w_orig, x + w)
        y2 = min(h_orig, y + h)

        face_img = image.crop((x, y, x2, y2))
        image = face_img

    # -------------------------------
    # Transform + Predict
    # -------------------------------
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    if return_face:
        return labels[pred], confidence, face_img

    return labels[pred], confidence