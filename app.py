import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# CONFIGURATION
# =========================
st.set_page_config(
    page_title="Cats vs Dogs AI",
    page_icon="🐱🐶",
    layout="centered"
)

# =========================
# CHARGEMENT DU MODELE
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cats_vs_dogs_cnn.h5")

model = load_model()

# IMPORTANT : doit être IDENTIQUE à ton entraînement
IMG_SIZE = (150, 150)

# =========================
# TITRE
# =========================
st.title("🐱🐶 Classification Cats vs Dogs")
st.write("Upload une image et le modèle va prédire s'il s'agit d'un chat ou d'un chien.")

# =========================
# UPLOAD IMAGE
# =========================
uploaded_file = st.file_uploader(
    "Choisis une image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PRÉTRAITEMENT
# =========================
def preprocess_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)

    # sécurité si image RGBA ou grayscale
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# =========================
# PREDICTION
# =========================
def predict(img_array):
    prediction = model.predict(img_array, verbose=0)[0][0]

    if prediction > 0.5:
        return "🐶 Chien", prediction
    else:
        return "🐱 Chat", prediction

# =========================
# APP LOGIC
# =========================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Image uploadée", use_column_width=True)

    with col2:
        st.write("🔄 Analyse en cours...")

        img_array = preprocess_image(image)
        label, score = predict(img_array)

        confidence = float(score if score > 0.5 else 1 - score)

        st.subheader("Résultat :")
        st.success(label)

        st.write(f"Confiance : **{confidence*100:.2f}%**")

        st.progress(confidence)

        st.bar_chart({
            "Chat": [1 - score],
            "Chien": [score]
        })

else:
    st.info("👆 Upload une image pour commencer la prédiction")