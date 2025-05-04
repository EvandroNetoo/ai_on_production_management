import cv2
import streamlit as st
import numpy as np
from keras.api.models import load_model

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Quality Assurance Demo", layout="wide")
st.title("🔍 Quality Assurance - Defect Detection")

# Sidebar: model and thresholds
model_path = st.sidebar.text_input("Modelo Keras (.h5)", "best_cube_model.h5")
threshold = st.sidebar.slider("Threshold de detecção", 0.0, 1.0, 0.5)

if st.sidebar.button("Carregar Modelo"):
    model = load_model(model_path)
    st.sidebar.success("Modelo carregado com sucesso!")
else:
    model = None

# Video feed placeholder
display = st.empty()

# Start webcam capture
cap = cv2.VideoCapture(0)

st.markdown("---")
st.write("Posicione o produto na frente da webcam para inspeção de qualidade.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Falha ao acessar a webcam.")
        break

    # Preprocess frame for model
    img = cv2.resize(frame, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    x_input = np.expand_dims(img_norm, axis=0)

    label = "Modelo não carregado"
    color = (0, 255, 0)

    if model:
        preds = model.predict(x_input)[0][0]
        if preds >= threshold:
            label = f"Defeito Detectado ({preds:.2f})"
            color = (0, 0, 255)
        else:
            label = f"OK ({preds:.2f})"
            color = (0, 255, 0)

    # Overlay result on frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    display.image(frame_display, use_container_width=True)

# Release resources
cap.release()
