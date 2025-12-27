import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# -----------------------------
# Load model
# -----------------------------
model = tf.keras.models.load_model("imdb_simple_rnn.h5")

# -----------------------------
# Parameters
# -----------------------------
vocab_size = 10000
maxlen = 100

# -----------------------------
# Load word index
# -----------------------------
word_index = imdb.get_word_index()

def encode_review(text):
    words = text.lower().split()
    encoded = []
    for word in words:
        if word in word_index and word_index[word] < vocab_size:
            encoded.append(word_index[word])
        else:
            encoded.append(2)  # unknown
    return encoded

# -----------------------------
# Sidebar (Professional Touch)
# -----------------------------
st.sidebar.title("📘 Project Info")
st.sidebar.write("""
**IMDb Sentiment Analysis**
- Model: Simple RNN
- Dataset: IMDb Movie Reviews
- Task: Binary Classification
- Output: Positive / Negative
""")

st.sidebar.markdown("---")
st.sidebar.write("👩‍💻 Built by **Koushali BS**")
st.sidebar.write("🚀 Deep Learning | NLP")

# -----------------------------
# Main UI
# -----------------------------
st.markdown("<h1 style='text-align: center;'>🎬 IMDb Movie Review Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a movie review and let the AI predict the sentiment</p>", unsafe_allow_html=True)

st.markdown("---")

review = st.text_area(
    "✍️ Enter Movie Review",
    height=180,
    placeholder="Example: This movie was absolutely amazing with great performances..."
)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔍 Analyze Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a movie review.")
    else:
        encoded = encode_review(review)
        padded = pad_sequences([encoded], maxlen=maxlen)

        prediction = model.predict(padded)
        prob = prediction[0][0]

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        if prob > 0.5:
            st.success("😊 **Positive Review**")
            st.progress(int(prob * 100))
            st.write(f"**Confidence:** {prob:.2f}")
        else:
            st.error("😠 **Negative Review**")
            st.progress(int((1 - prob) * 100))
            st.write(f"**Confidence:** {1 - prob:.2f}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 13px;'>Built using TensorFlow, Keras & Streamlit</p>",
    unsafe_allow_html=True
)
