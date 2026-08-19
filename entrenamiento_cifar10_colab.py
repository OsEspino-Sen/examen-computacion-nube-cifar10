# ============================================================
# ENTRENAMIENTO CIFAR-10 | GOOGLE COLAB
# ============================================================
# Ejecuta este archivo por celdas en Google Colab o utiliza
# el notebook entrenamiento_cifar10_colab.ipynb.
#
# Resultado final:
#   models/cifar10_model.keras
# ============================================================

import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)

# 1) Dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

print("Train:", x_train.shape, y_train.shape)
print("Test :", x_test.shape, y_test.shape)

# 2) Normalización y etiquetas
# El modelo incluye Rescaling para que la misma lógica exista
# durante entrenamiento e inferencia.
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 128

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(10000)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# 3) CNN básica, pero suficientemente sólida para el examen
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(0.08, 0.08),
    ],
    name="data_augmentation",
)

model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        data_augmentation,
        layers.Rescaling(1.0 / 255),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.20),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.30),
        layers.Dense(10, activation="softmax"),
    ],
    name="cifar10_cnn",
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# 4) Callbacks para detenerse cuando deje de mejorar
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        mode="max",
        restore_best_weights=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]

# 5) Entrenamiento
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=20,
    callbacks=callbacks,
)

# 6) Evaluación
loss, accuracy = model.evaluate(test_ds, verbose=0)
print(f"Accuracy final: {accuracy:.2%}")

# 7) Guardado en formato Keras recomendado
os.makedirs("models", exist_ok=True)
model.save("models/cifar10_model.keras")
print("Modelo guardado en models/cifar10_model.keras")

# 8) Descargar el archivo desde Colab
from google.colab import files
files.download("models/cifar10_model.keras")
