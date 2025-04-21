import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

@st.cache(allow_output_mutation=True)
def load_your_model():
    model = load_model('aidetect.h5')
    return model

model = load_your_model()

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image_array = np.array(image)
    
    if image_array.ndim == 2:
        image_array = np.stack((image_array,)*3, axis=-1)

    image_array = image_array.astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

st.title("AI vs Human Image Detector")
st.write("Upload an image to detect whether it is AI generated or taken by a human.")

uploaded_file = st.file_uploader("Choose an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    processed_image = preprocess_image(image)
    prediction_prob = model.predict(processed_image)
    
    if prediction_prob[0] < 0.5:
        prediction = "Human generated"
    else:
        prediction = "AI generated"

    st.success(f"Prediction: The image is {prediction}.")