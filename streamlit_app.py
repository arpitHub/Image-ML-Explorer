import streamlit as st

# --- App Title ---
st.set_page_config(page_title="Image Explorer App", page_icon="🖼️")

st.title("🖼️ Image Explorer App")
st.markdown("""
Welcome to the **Image Explorer App** — an interactive tool for learning how machine learning models
work with image data.

This app is designed to guide students step by step:
1. **Dataset Explorer (📂)** – Load and preview image datasets.
2. **Preprocessing (🖼️)** – Normalize and reshape images for modeling.
3. **Model Builder (🤖)** – Train classic ML models (Logistic Regression, kNN).
4. **Results (📊)** – Test predictions and explore misclassifications.
5. **CNN Model (🧠)** – Train a Convolutional Neural Network.
6. **CNN Results (📊)** – Evaluate CNN predictions and confidence scores.

👉 Use the sidebar to navigate between pages.
""")

# --- Teaching Note ---
st.write("### Learning Outcomes")
st.markdown("""
By the end of this app, you will:
- Understand how images are represented as pixel arrays.
- Learn preprocessing techniques like normalization and reshaping.
- Compare classic ML models with deep learning (CNNs).
- Interpret confusion matrices, probability scores, and misclassified examples.
""")

st.info("Start with the **Dataset Explorer** page to load and preview images.")
