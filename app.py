import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Charger le modèle
model = tf.keras.models.load_model("efficientnet_model.keras")  # à adapter à ton chemin

# Noms des classes
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

st.title("🌍 Classification d'images de paysages (EfficientNetB0)")

uploaded_file = st.file_uploader("📤 Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image sélectionnée", use_column_width=True)

    # Prétraitement
    img = image.resize((150, 150))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    img_array = preprocess_input(img_array)

    # Prédiction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.markdown(f"### ✅ Prédiction : **{predicted_class}**")
