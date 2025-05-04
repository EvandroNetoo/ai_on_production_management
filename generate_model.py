import tensorflow as tf
from keras import layers
# 1) Carregamento dos datasets: substitua pelas suas pastas
train_ds = tf.keras.utils.image_dataset_from_directory(
    'data/train',            # ← seu diretório de treino
    labels='inferred',
    label_mode='int',
    class_names=['solved','unsolved'],  # ← suas duas classes
    image_size=(240, 240),
    batch_size=32)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'data/valid',            # ← seu diretório de validação
    labels='inferred',
    label_mode='int',
    class_names=['solved','unsolved'],
    image_size=(240, 240),
    batch_size=32)

test_ds = tf.keras.utils.image_dataset_from_directory(
    'data/test',             # ← seu diretório de teste
    labels='inferred',
    label_mode='int',
    class_names=['solved','unsolved'],
    image_size=(240, 240),
    batch_size=32)

# 2) Ajuste do número de classes na última camada do modelo
num_classes = 2  # ← apenas duas classes: solved vs unsolved

# Exemplo no bloco de construção do modelo:
model = tf.keras.Sequential([
    layers.Rescaling(1./255, input_shape=(240, 240, 3)),
    # ... (camadas intermediárias do tutorial) ...
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)           # ← alteração aqui
])

# 3) Escolha da função de perda adequada
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # ← mantém 'from_logits=True'
    metrics=['accuracy']
)

# 4) Treinamento
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10           # ← ajuste o nº de épocas conforme testes
)

model.save('rubiks_model.h5')  