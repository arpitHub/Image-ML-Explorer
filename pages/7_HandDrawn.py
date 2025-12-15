import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.title("✍️ Hand-Drawn Image Tester")

# Check if models exist
if "best_model" not in st.session_state and "cnn_model" not in st.session_state:
    st.warning("Please train a model first in the Model Builder or CNN Model page.")
else:
    st.write("### Draw a digit (0–9) below and let the model predict it!")

    # --- Drawing canvas ---
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        width=128,
        height=128,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        img = canvas_result.image_data[:, :, 0]  # grayscale channel

        # --- Skip empty drawings ---
        if np.sum(img) == 0:
            st.info("Draw a digit above to see predictions.")
        else:
            # --- Resize to 8x8 like Digits dataset ---
            img_resized = Image.fromarray(img).resize((8, 8))
            img_array = np.array(img_resized)

            # Invert colors (Digits dataset is white digits on black background)
            img_array = 255 - img_array

            # Scale to 0–16 range (Digits dataset pixel values)
            img_scaled = (img_array / 255.0) * 16.0

            # --- Show resized drawing ---
            st.write("### Your Drawing (Resized to 8×8)")
            fig, ax = plt.subplots()
            ax.imshow(img_scaled, cmap="gray")
            ax.axis("off")
            st.pyplot(fig)

            # --- Classic ML Prediction ---
            if "best_model" in st.session_state:
                model = st.session_state["best_model"]
                img_flattened = img_scaled.flatten().reshape(1, -1)
                pred = model.predict(img_flattened)
                st.success(f"Classic ML Prediction: {pred[0]}")

            # --- CNN Prediction ---
            if "cnn_model" in st.session_state:
                cnn_model = st.session_state["cnn_model"]
                img_cnn = img_scaled.reshape(1, 8, 8, 1)
                pred_probs = cnn_model.predict(img_cnn)
                pred_label = np.argmax(pred_probs)
                st.success(f"CNN Prediction: {pred_label}")
                st.write("### Probability Scores")
                st.bar_chart(pred_probs[0])

            # --- Teaching Note ---
            st.write("### Teaching Note")
            st.markdown("""
            Hand‑drawn inputs often look different from dataset images:

            - **Resolution mismatch:** The digits dataset uses small 8×8 grayscale images, while drawings may have thicker strokes.
            - **Noise & variation:** Human sketches introduce irregularities not seen in the clean dataset.
            - **Model limitations:** Classic ML models struggle with these differences, while CNNs handle them better by learning filters.

            👉 This shows why preprocessing and robust models are critical when moving from curated datasets to real‑world inputs.
            """)
