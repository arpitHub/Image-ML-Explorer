import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import numpy as np

st.title("📂 Image Dataset Explorer")

# --- Explanations ---
st.write("### How Image Datasets Work")
st.markdown("""
Images are stored as arrays of pixel values.  
For example, a grayscale image is a 2D matrix where each number represents brightness (0 = black, 255 = white).

👉 Machine learning models learn patterns in these numbers to classify images.
""")

# --- Load default dataset (Digits) ---
digits = load_digits()

X = digits.images   # shape (n_samples, 8, 8)
y = digits.target   # labels (0–9)

st.write("Dataset shape:", X.shape)
st.write("Number of classes:", len(np.unique(y)))

# --- Show sample images ---
st.write("### Sample Images")
num_samples = st.slider("Select number of samples to view:", 4, 20, 8)

fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
for i, ax in enumerate(axes):
    ax.imshow(X[i], cmap="gray")
    ax.set_title(f"Label: {y[i]}")
    ax.axis("off")
st.pyplot(fig)

# --- Save for later pages ---
st.session_state["X_images"] = X
st.session_state["y_labels"] = y
