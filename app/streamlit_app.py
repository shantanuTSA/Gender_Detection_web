import streamlit as st
from PIL import Image
from src.predict import predict_image

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Gender Detection", layout="centered")

st.title("🧠 Gender Detection App")
st.markdown("Upload an image or use your webcam to predict gender.")

st.divider()

# -------------------------------
# Upload Section
# -------------------------------
st.header("📁 Upload Image")

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict (Upload)"):
        with st.spinner("Predicting..."):
            pred, conf = predict_image(image)

        st.success(f"Prediction: {pred.upper()}")
        st.info(f"Confidence: {conf:.2f}")

st.divider()

# -------------------------------
# Webcam Section
# -------------------------------
st.header("📸 Webcam")

camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    image = Image.open(camera_image).convert("RGB")
    st.image(image, caption="Captured Image", use_container_width=True)

    if st.button("Predict (Camera)"):
        with st.spinner("Predicting..."):
            pred, conf = predict_image(image)

        st.success(f"Prediction: {pred.upper()}")
        st.info(f"Confidence: {conf:.2f}")

st.divider()

st.caption("Built with PyTorch + Streamlit + OpenCV")