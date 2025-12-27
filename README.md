# 🎬 IMDb Movie Review Sentiment Analysis using Simple RNN

This project is an **end-to-end Natural Language Processing (NLP) application** that predicts whether a movie review is **Positive** or **Negative** using a **Simple Recurrent Neural Network (RNN)**.

The model is trained on the **IMDb Movie Reviews dataset** and deployed as an interactive **Streamlit web application**.

---

## 🚀 Live Demo
👉 *(Add your Streamlit Cloud link here after deployment)*

---

## 📊 Dataset
- **Dataset Name:** IMDb Movie Reviews
- **Source:** TensorFlow / Keras built-in dataset
- **Total Reviews:** 50,000
- **Training Samples:** 25,000
- **Testing Samples:** 25,000

### Labels
- `1` → Positive Review  
- `0` → Negative Review  

---

## 🧠 Model Architecture
The project uses a **Simple RNN** with the following layers:

- **Embedding Layer**
  - Vocabulary size: 10,000
  - Embedding dimension: 128
- **Simple RNN Layer**
  - 128 neurons
- **Dense Output Layer**
  - 1 neuron
  - Sigmoid activation

### Why RNN?
RNNs are suitable for **sequential text data**, where the order of words matters for understanding sentiment.

---

## ⚙️ Technologies Used
- Python 3
- TensorFlow / Keras
- NumPy
- Streamlit

---

## 📈 Model Performance
- **Test Accuracy:** ~75–78%
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam

> Note: Simple RNNs have limitations in capturing long-term dependencies. More advanced architectures like **LSTM** or
