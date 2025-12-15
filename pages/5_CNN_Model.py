import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

st.title("🧠 Convolutional Neural Network (CNN)")

if "X_images" not in st.session_state or "y_labels" not in st.session_state:
    st.warning("Please explore the dataset first in the Dataset Explorer page.")
else:
    X = st.session_state["X_images"]
    y = st.session_state["y_labels"]

    st.write("### Why CNNs?")
    st.markdown("""
    Convolutional Neural Networks (CNNs) are designed for image data.  
    They learn **filters** that detect edges, shapes, and textures, building up to complex patterns.  
    👉 This makes them far more powerful than models that treat images as flat vectors.
    """)

    # --- Preprocess data for CNN ---
    X = X / 16.0  # normalize pixel values (digits dataset ranges 0–16)
    X = X[..., np.newaxis]  # add channel dimension
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Build CNN model ---
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(8, 8, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # --- Train model ---
    st.write("### Training CNN")
    epochs = st.slider("Select number of epochs:", 1, 20, 5)
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=0)

    # --- Evaluate ---
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    st.success(f"CNN Test Accuracy: {test_acc:.2f}")

    # --- Plot training history ---
    st.write("### Training History")
    fig, ax = plt.subplots()
    ax.plot(history.history["accuracy"], label="Train Accuracy")
    ax.plot(history.history["val_accuracy"], label="Validation Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()
    st.pyplot(fig)

    # --- Save CNN model ---
    st.session_state["cnn_model"] = model
