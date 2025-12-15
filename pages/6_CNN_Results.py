import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.title("📊 CNN Results & Predictions")

if "cnn_model" not in st.session_state or "X_images" not in st.session_state:
    st.warning("Please train a CNN first in the CNN Model page.")
else:
    model = st.session_state["cnn_model"]
    X = st.session_state["X_images"]
    y = st.session_state["y_labels"]

    # Normalize and reshape for CNN input
    X_norm = X / 16.0
    X_norm = X_norm[..., np.newaxis]

    st.write("### Test CNN on New Images")
    idx = st.slider("Choose an image index:", 0, len(X_norm) - 1, 0)

    # Show the image
    fig, ax = plt.subplots()
    ax.imshow(X[idx], cmap="gray")
    ax.axis("off")
    st.pyplot(fig)

    # Prediction
    pred_probs = model.predict(np.expand_dims(X_norm[idx], axis=0))
    pred_label = np.argmax(pred_probs)

    st.write(f"**Predicted Label:** {pred_label}")
    st.write(f"**True Label:** {y[idx]}")

    # Probability scores
    st.write("### Probability Scores")
    st.bar_chart(pred_probs[0])
    st.markdown("👉 Higher probability = more confidence in the prediction.")

    # Misclassified examples
    st.write("### Explore Misclassified Images")
    preds = np.argmax(model.predict(X_norm), axis=1)
    misclassified = np.where(preds != y)[0]

    if len(misclassified) > 0:
        sample_idx = st.selectbox("Choose a misclassified index:", misclassified)
        fig, ax = plt.subplots()
        ax.imshow(X[sample_idx], cmap="gray")
        ax.set_title(f"True: {y[sample_idx]}, Predicted: {preds[sample_idx]}")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.success("🎉 No misclassified examples found — the CNN predicted all correctly!")
