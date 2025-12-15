import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.title("📊 Results & Predictions")

if "best_model" not in st.session_state or "X_processed" not in st.session_state:
    st.warning("Please train a model first in the Model Builder page.")
else:
    best_model = st.session_state["best_model"]
    X = st.session_state["X_processed"]
    y = st.session_state["y_labels"]

    st.write("### Test the Model on New Images")
    st.markdown("""
    You can select an image from the dataset and see how the model predicts it.
    """)

    # --- Select an image index ---
    idx = st.slider("Choose an image index:", 0, len(X) - 1, 0)

    # --- Show the image ---
    fig, ax = plt.subplots()
    if len(X.shape) == 2:  # flattened
        side = int(np.sqrt(X.shape[1]))
        ax.imshow(X[idx].reshape(side, side), cmap="gray")
    else:  # 2D images
        ax.imshow(X[idx], cmap="gray")
    ax.axis("off")
    st.pyplot(fig)

    # --- Prediction ---
    pred = best_model.predict([X[idx].flatten()]) if len(X.shape) == 2 else best_model.predict([X[idx].ravel()])
    st.write(f"**Predicted Label:** {pred[0]}")
    st.write(f"**True Label:** {y[idx]}")

    # --- Probability scores (if available) ---
    if hasattr(best_model, "predict_proba"):
        probs = best_model.predict_proba([X[idx].flatten()]) if len(X.shape) == 2 else best_model.predict_proba([X[idx].ravel()])
        st.write("### Probability Scores")
        st.bar_chart(probs[0])
        st.markdown("👉 Higher probability = more confidence in the prediction.")

    # --- Misclassified examples ---
    st.write("### Explore Misclassified Images")
    st.markdown("Sometimes models make mistakes. Let's look at a few misclassified examples.")

    preds = best_model.predict(X.reshape(len(X), -1)) if len(X.shape) == 2 else best_model.predict(X.reshape(len(X), -1))
    misclassified = np.where(preds != y)[0]

    if len(misclassified) > 0:
        sample_idx = st.selectbox("Choose a misclassified index:", misclassified)
        fig, ax = plt.subplots()
        if len(X.shape) == 2:
            side = int(np.sqrt(X.shape[1]))
            ax.imshow(X[sample_idx].reshape(side, side), cmap="gray")
        else:
            ax.imshow(X[sample_idx], cmap="gray")
        ax.set_title(f"True: {y[sample_idx]}, Predicted: {preds[sample_idx]}")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.success("🎉 No misclassified examples found — the model predicted all correctly!")
