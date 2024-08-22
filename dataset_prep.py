from keras.utils import image_dataset_from_directory
import tensorflow as tf

train_data = image_dataset_from_directory(directory="../../DATASET/lung_collapse/train", shuffle=True,
                                          batch_size=32, image_size=(256, 256),
                                          color_mode="grayscale", validation_split=0.2, subset="training",
                                          seed=42)

val_data = image_dataset_from_directory(directory="../../DATASET/lung_collapse/train", shuffle=True,
                                        batch_size=32, image_size=(256, 256),
                                        color_mode="grayscale", validation_split=0.2, subset="validation",
                                        seed=42)
print(train_data.class_names)

def process(image, label):
    image = tf.cast(image / 255., tf.float32)
    return image, label


train_data = train_data.map(process)
val_data = val_data.map(process)


