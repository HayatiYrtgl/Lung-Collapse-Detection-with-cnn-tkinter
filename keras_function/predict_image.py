import keras
import numpy as np
from keras.models import load_model


def predict_lung_collapse(image_tensor: np.array, model: keras.Model):
    prediction = model.predict(image_tensor)
    return f"{prediction[0][0] * 100}"
