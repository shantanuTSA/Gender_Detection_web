from flask import Flask, request, jsonify
from PIL import Image
import io

from src.predict import predict_image

app = Flask(__name__)

@app.route("/")
def home():
    return "Gender Detection API Running"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        image = Image.open(file.stream).convert("RGB")
        pred, conf = predict_image(image)

        return jsonify({
            "prediction": pred,
            "confidence": float(conf)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)