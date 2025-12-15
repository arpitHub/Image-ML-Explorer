import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🤖 Model Builder")

if "X_processed" not in st.session_state or "y_labels" not in st.session_state:
    st.warning("Please preprocess the dataset first in the Preprocessing page.")
else:
    X = st.session_state["X_processed"]
    y = st.session_state["y_labels"]

    st.write("### How Models Classify Images")
    st.markdown("""
    Once images are converted into numbers, we can train machine learning models:

    - **Logistic Regression:** Learns weights for each pixel to separate classes.
    - **k-Nearest Neighbors (kNN):** Compares new images to the closest training examples.
    - **Convolutional Neural Network (CNN):** Learns filters that detect edges, shapes, and textures (advanced).

    👉 Each model has strengths: Logistic Regression is simple, kNN is intuitive, CNNs are powerful for complex images.
    """)

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        # CNN placeholder (to be added later with TensorFlow/PyTorch)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = (model, acc, preds)

    # --- Display results ---
    st.write("### Model Performance")
    for name, (model, acc, preds) in results.items():
        st.write(f"**{name}** → Accuracy: {acc:.2f}")

    # --- Best model ---
    best_model_name = max(results, key=lambda k: results[k][1])
    best_model, best_acc, best_preds = results[best_model_name]
    st.success(f"Best model: {best_model_name} (Accuracy: {best_acc:.2f})")

    st.session_state["best_model"] = best_model

    # --- Confusion Matrix ---
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, best_preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
