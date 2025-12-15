import streamlit as st
import numpy as np

st.title("🖼️ Image Preprocessing")

if "X_images" not in st.session_state or "y_labels" not in st.session_state:
    st.warning("Please explore the dataset first in the Dataset Explorer page.")
else:
    X = st.session_state["X_images"]
    y = st.session_state["y_labels"]

    st.write("### Why Preprocessing Matters")
    st.markdown("""
    Raw images are just arrays of pixel values.  
    Preprocessing ensures they are consistent and easier for models to learn from.

    Common steps:
    - **Normalization:** Scale pixel values from 0–255 down to 0–1.
    - **Reshaping:** Flatten images into vectors or keep them as 2D arrays.
    - **Grayscale/Color Conversion:** Simplify images if needed.
    """)

    # --- Normalization ---
    st.write("### Normalization")
    st.markdown("Pixel values range from 0–255. Normalizing scales them to 0–1.")
    X_normalized = X / 16.0  # Digits dataset pixels range 0–16
    st.write("Example normalized pixel values (first image):")
    st.write(X_normalized[0])

    # --- Flatten vs Keep 2D ---
    st.write("### Reshaping Options")
    option = st.radio("Choose how to reshape images:", ["Flatten (vector)", "Keep 2D"])

    if option == "Flatten (vector)":
        X_processed = X_normalized.reshape(len(X_normalized), -1)
        st.write("Shape after flattening:", X_processed.shape)
    else:
        X_processed = X_normalized
        st.write("Shape (2D images):", X_processed.shape)

    # --- Save for later pages ---
    st.session_state["X_processed"] = X_processed
    st.session_state["y_labels"] = y
