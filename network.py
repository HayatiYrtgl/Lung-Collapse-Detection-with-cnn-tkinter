from keras.layers import *
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from dataset_prep import train_data, val_data

# network
def create_cnn():
    model = Sequential(name="lung_collapse")
    model.add(Input(shape=(256, 256, 1)))
    model.add(Conv2D(filters=32, kernel_size=5, strides=1, padding="valid"))
    model.add(Conv2D(filters=32, kernel_size=5, strides=1, padding="valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(filters=64, kernel_size=5, strides=1, padding="valid"))
    model.add(Conv2D(filters=64, kernel_size=5, strides=1, padding="valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(filters=128, kernel_size=5, strides=1, padding="valid"))
    model.add(Conv2D(filters=128, kernel_size=5, strides=1, padding="valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(filters=256, kernel_size=5, strides=1, padding="valid"))
    model.add(Conv2D(filters=256, kernel_size=5, strides=1, padding="valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(filters=512, kernel_size=5, strides=1, padding="valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling2D(2, 2))

    model.add(Flatten())

    model.add(Dense(2048, activation="relu"))
    model.add(Dropout(0.4))

    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.4))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=["accuracy"])

    model.summary()

    return model

model = create_cnn()

model.fit(train_data, validation_data=val_data, callbacks=[ModelCheckpoint(filepath="lung_model/lung.h5", monitor="val_loss",
                                                                           mode="min")],
          epochs=30)

