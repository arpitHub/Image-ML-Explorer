# 🖼️ Image ML Explorer App

An interactive **Streamlit app** for learning how machine learning models process and classify image data.  
Students can explore datasets, preprocess images, build models, and visualize results — all with clear explanations and examples.

---

## 🚀 Features

- **Dataset Explorer (📂)**  
  Preview image datasets (e.g., MNIST digits), inspect samples, and understand how images are stored as pixel arrays.

- **Preprocessing (🖼️)**  
  - Normalize pixel values (0–255 → 0–1).  
  - Flatten images into vectors or keep 2D structure.  
  - Teaching notes explaining why preprocessing matters.

- **Model Builder (🤖)**  
  - Train Logistic Regression and k‑Nearest Neighbors.  
  - Compare accuracy across models.  
  - Confusion matrix visualization.  
  - Teaching notes on strengths and limitations of each model.

- **Results (📊)**  
  - Test predictions on new images.  
  - Probability scores for confidence.  
  - Explore misclassified examples to understand model errors.

- **CNN Model (🧠)**  
  - Build and train a Convolutional Neural Network (CNN).  
  - Visualize training vs validation accuracy.  
  - Compare CNN performance with classic ML models.

- **CNN Results (📊)**  
  - Test CNN predictions with probability scores.  
  - Explore misclassified examples to highlight CNN strengths and weaknesses.
 

- **Hand‑Drawn Image Tester (✍️)**  
  - Draw your own digit directly in the app using an interactive canvas.
  - See how both classic ML models and CNNs interpret your sketch.
  - Learn why hand‑drawn inputs differ from curated datasets (noise, resolution mismatch, thicker strokes).
  - Reinforces the importance of preprocessing and robust models in real‑world scenarios.
---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) for interactive UI  
- [scikit-learn](https://scikit-learn.org/) for classic ML models  
- [TensorFlow/Keras](https://www.tensorflow.org/) for CNNs  
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) for plots  

---

## 📂 Project Structure

```text
├── streamlit_app.py            # Main entry point and landing page
├── pages/
│   ├── 1_Dataset_Explorer.py  # Load and preview image datasets
│   ├── 2_Preprocessing.py     # Normalize and reshape images
│   ├── 3_Model_Builder.py     # Train Logistic Regression and kNN models
│   ├── 4_Results.py           # Test predictions and explore misclassifications
│   ├── 5_CNN_Model.py         # Build and train a Convolutional Neural Network
│   ├── 6_CNN_Results.py       # Evaluate CNN predictions and confidence scores
│   └── 7_HandDrawn.py         # Interactive canvas for drawing digits and testing models
├── requirements.txt           # Dependencies with pinned versions
├── runtime.txt                # Python version specification
└── README.md                  # Project guide and documentation
```

---

## ⚡ How to Run Locally

1. Clone the repo:
   ```bash
   $ git clone https://github.com/your-username/image-explorer-app.git
   $ cd image-explorer-app
   ```

2. Install dependencies:
```bash
$ pip install -r requirements.txt
```

3. Launch the app:
```bash
$ streamlit run streamlit_app.py
```
---

## 🌐 Deployment
Push your repo to GitHub.

Go to Streamlit Cloud.

Connect your repo and select streamlit_app.py as the entry point.

Deploy and share the link with students!

Example URL: https://image-explorer.streamlit.app

---

## 🎓 Learning Outcomes
By using this app, students will:

Understand how images are represented as pixel arrays.

Learn preprocessing techniques like normalization and reshaping.

Compare classic ML models with deep learning (CNNs).

Interpret confusion matrices, probability scores, and misclassified examples.

Appreciate the strengths and limitations of different approaches to image classification.

---

## 📸 Screenshots (optional)
Add screenshots of each page once deployed.

---

## ✅ Notes
- `streamlit_app.py` introduces the app and guides navigation.  
-  Each file in `pages/` corresponds to a learning module.  
- `requirements.txt` + `runtime.txt` ensure reproducible deployment.  
- `README.md` provides instructions, features, and learning outcomes.  

---

🙌 Credits
Built with ❤️ by Arpit to make machine learning hands‑on and approachable for students.

