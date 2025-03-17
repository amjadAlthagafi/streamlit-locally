import streamlit as st
import tensorflow as tf
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gensim.downloader as api

# Load pre-trained Word2Vec model
w2v_model = api.load("word2vec-google-news-300")
embedding_dim = 300  # Word vector size
max_length = 30  # Fixed sentence length

# Function to convert text into a list of word vectors
def text_to_sequence(text, model):
    words = word_tokenize(text.lower())
    return [model[word] for word in words if word in model]

# Load trained CNN model
model = tf.keras.models.load_model("cnn_model_headline.h5")

# Streamlit app
st.title("News Headline Classification App")
st.write("Enter a news headline to classify it using the trained CNN model.")

headline = st.text_input("Enter a headline:")

if st.button("Predict"):
    if headline:
        # Convert input text to sequence
        sequence = text_to_sequence(headline, w2v_model)
        padded_sequence = pad_sequences([sequence], maxlen=max_length, dtype='float32', padding='post', truncating='post', value=0.0)
        
        # Predict
        prediction = model.predict(padded_sequence)[0][0]
        label = "Positive" if prediction > 0.7 else "Negative"
        
        st.write(f"**Prediction:** {label} (Confidence: {prediction:.4f})")
    else:
        st.write("Please enter a headline.")
