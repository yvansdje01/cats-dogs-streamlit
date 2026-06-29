import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Cats vs Dogs CNN",
    page_icon="🐱🐶",
    layout="centered"
)

IMG_SIZE = (150, 150)

# ---------------- LOAD MODEL (SAFE CACHE) ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cats_vs_dogs_cnn.keras")

model = load_model()

# ---------------- TITLE ----------------
st.title("🐱🐶 Cats vs Dogs Classifier")
st.write("Upload une image et le modèle va prédire Chat ou Chien.")

# ---------------- UPLOAD IMAGE ----------------
uploaded_file = st.file_uploader(
    "Choisis une image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREPROCESS ----------------
def preprocess(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ---------------- PREDICTION ----------------
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Image uploadée", use_container_width=True)

        img_array = preprocess(image)

        prediction = model.predict(img_array)[0][0]

        # Interprétation
        if prediction > 0.5:
            label = "🐶 Chien"
            confidence = prediction
        else:
            label = "🐱 Chat"
            confidence = 1 - prediction

        st.subheader("Résultat")
        st.write(f"**Classe :** {label}")
        st.write(f"**Confiance :** {confidence:.2f}")

        # Barre de confiance
        st.progress(float(confidence))

    except Exception as e:
        st.error("❌ Erreur lors du traitement de l'image")
        st.write(e)