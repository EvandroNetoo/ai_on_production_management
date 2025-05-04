from keras.api.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import layers
from keras import models

model = load_model('rubiks_model.h5')

img = keras.utils.load_img("data/test/solved/v2_20220306_150817_jpg.rf.b3de3fa6f635e581f906c2d2311e7a59.jpg", target_size=(240, 240))
plt.imshow(img)

img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(keras.ops.sigmoid(predictions[0][0]))
print(score)
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
