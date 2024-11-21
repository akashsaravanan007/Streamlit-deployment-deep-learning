import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import get_file
import numpy as np

# Step 1: Download and Prepare the Dataset
# Download the dataset manually
url = "https://storage.googleapis.com/download.tensorflow.org/example_images/beans.zip"
dataset_path = get_file("streamlit/beans.zip", origin=url, extract=True)

# Extracted data directory
data_dir = os.path.join(os.path.dirname(dataset_path), "beans")

# Load training and validation datasets
IMG_SIZE = 224
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
)

# Normalize pixel values
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(preprocess).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Visualize some images from the training dataset
def visualize_images(dataset, n=5):
    plt.figure(figsize=(10, 10))
    for i, (image, label) in enumerate(dataset.take(n)):
        plt.subplot(1, n, i + 1)
        plt.imshow(image.numpy())
        plt.title(f"Class: {label.numpy()}")
        plt.axis("off")
    plt.show()

visualize_images(train_ds)

# Step 2: Build the Model
# Load MobileNetV2 with pre-trained weights
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze base model layers

# Add custom classification layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(3, activation="softmax")  # 3 classes in the beans dataset
])

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Step 3: Train the Model
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stopping]
)

# Step 4: Evaluate the Model
results = model.evaluate(val_ds, return_dict=True)
print(f"Validation Accuracy: {results['accuracy']:.2f}")
print(f"Validation Loss: {results['loss']:.2f}")

# Step 5: Save the Model
model.save("../streamlit/beans_model.keras")
print("Model saved as beans_model.keras")
