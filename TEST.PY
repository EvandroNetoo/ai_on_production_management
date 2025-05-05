import cv2                                  # OpenCV para captura de vídeo :contentReference[oaicite:0]{index=0}
import numpy as np
from tensorflow.keras.models import load_model  # Carrega modelo Keras :contentReference[oaicite:1]{index=1}
from tensorflow import keras

# 1) Load do modelo treinado
model = load_model('best_cube_model.h5')

# 2) Inicializa captura da webcam (dispositivo 0)
cap = cv2.VideoCapture(0)

# 3) Loop de captura e predição
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3.1) Pre-processamento: redimensiona e normaliza
    img = cv2.resize(frame, (224, 224))
    img_array = np.expand_dims(img.astype('float32') / 255.0, axis=0)

    # 3.2) Predição
    # preds = model.predict(img_array)
    # print(float(keras.ops.sigmoid(preds[0][0])))
    # class_idx = np.argmax(preds, axis=1)[0]
    # label = 'Solved' if class_idx < 0.5 else 'Unsolved'  # 0=solved, 1=unsolved :contentReference[oaicite:2]{index=2}
    
    pred = model.predict(img_array)[0][0]
    label = ('Solved' if pred < 0.5 else 'Unsolved') + f' ({pred:.2f})'
    color = (0,255,0) if pred < 0.5 else (0,0,255)

    # 3.3) Exibe resultado na janela
    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Rubiks Classifier', frame)
    cv2.imshow('IMG', img)

    # Encerra ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4) Libera recursos
cap.release()
cv2.destroyAllWindows()
