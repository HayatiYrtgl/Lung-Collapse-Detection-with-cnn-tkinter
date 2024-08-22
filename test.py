from keras.utils import image_dataset_from_directory
import tensorflow as tf
from keras.models import load_model


def preprocess(image, label):
    img = tf.cast(image / 255.0, tf.float32)
    return img, label

test = image_dataset_from_directory(directory="../../DATASET/lung_collapse/test", image_size=(256, 256),
                                    color_mode="grayscale")
test = test.map(preprocess)

model = load_model("lung_model/lung.h5")
model.evaluate(test)


