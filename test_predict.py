from src.predict import predict_image
from PIL import Image

img = Image.open("Data/test/men/143.jpg")

pred, conf, face = predict_image(img, return_face=True)

print(pred, conf)

if face:
    face.save("face_output.jpg")
    print("Face saved!")
else:
    print("No face detected")