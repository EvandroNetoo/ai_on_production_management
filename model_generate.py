import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint

# --- 1. Gerar ImageDataGenerators ---
base_dir = 'data'
img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    base_dir + '/train',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    base_dir + '/valid',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    base_dir + '/test',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# --- 2. Definir a arquitetura do modelo ---
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=img_size + (3,)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# --- 3. Compilar ---
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- 4. Checkpoint para salvar o melhor modelo ---
checkpoint = ModelCheckpoint(
    'best_cube_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# --- 5. Treinar ---
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    callbacks=[checkpoint]
)

# --- 6. Avaliar no conjunto de teste ---
model.load_weights('best_cube_model.h5')
test_loss, test_acc = model.evaluate(test_gen)
print(f'Test accuracy: {test_acc:.4f}')
