# 🧠 Gender Detection Web App

🔗 **Live Demo:** https://genderdetectionweb-vtxs4qbgwyglqmmdnxxkpe.streamlit.app/

---

## 🚀 Overview

This project is an end-to-end deep learning application that predicts gender (**Male/Female**) from images. It combines computer vision, deep learning, and web deployment into a single pipeline.

The system supports both:

* 📁 Image upload
* 📸 Real-time webcam input

To improve prediction quality, the app first detects and crops faces before passing them to the model.

---

## ✨ Features

* 🔍 Face detection using OpenCV (Haar Cascade)
* 🧠 Deep learning model (EfficientNet-based)
* 📸 Webcam-based real-time prediction
* 📁 Image upload support
* 📊 Confidence score output
* 🌐 Deployed on Streamlit Cloud

---

## 🏗️ Architecture

```
User Input (Upload / Webcam)
        ↓
Face Detection (OpenCV)
        ↓
Image Preprocessing
        ↓
Deep Learning Model (PyTorch)
        ↓
Prediction (Male / Female + Confidence)
        ↓
Streamlit UI
```

---

## 🛠️ Tech Stack

* **Deep Learning:** PyTorch, Torchvision
* **Computer Vision:** OpenCV
* **Frontend/UI:** Streamlit
* **Data Handling:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn

---

## 📁 Project Structure

```
Gender_Detection_web/
│
├── app/
│   └── streamlit_app.py      # Streamlit UI
│
├── src/
│   ├── predict.py            # Model + inference logic
│   └── haarcascade_frontalface_default.xml
│
├── models/
│   └── best.pth              # Trained model
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup (Local)

```bash
# Clone the repo
git clone https://github.com/shantanuTSA/Gender_Detection_web.git
cd Gender_Detection_web

# Create virtual environment
python3 -m venv myenv
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app/streamlit_app.py
```

---

## 🧪 Usage

1. Open the app
2. Upload an image OR use webcam
3. Click predict
4. View:

   * Predicted gender
   * Confidence score

---

## 📊 Model Details

* Architecture: EfficientNet (custom classifier head)
* Input Size: 224 × 224
* Output: 2 classes (Male, Female)
* Loss Function: Cross Entropy Loss
* Optimizer: Adam

---

## ⚠️ Limitations

* Performance depends on:

  * Lighting conditions
  * Face clarity
  * Image angle
* Haar Cascade face detection is basic and may fail in complex scenes
* Model accuracy (~84%) can be improved with larger datasets

---

## 🚀 Future Improvements

* 🔥 Replace Haar Cascade with MediaPipe / MTCNN
* 🎯 Improve model accuracy with better dataset
* 📦 Optimize model size for faster inference
* 🎨 Enhance UI (bounding boxes, overlays)

---

## 👨‍💻 Author

**Shantanu**
IIIT Bangalore

---

## ⭐ Acknowledgements

* PyTorch & Torchvision
* OpenCV
* Streamlit

---

## 📌 Note

This project demonstrates a complete machine learning pipeline from model training to deployment as an interactive web application.
