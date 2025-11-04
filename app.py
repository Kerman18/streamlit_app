import os
import streamlit as st
import torch
from utils import NextTokenPredictor, generate_next_words
import os

MODELS_DIR = "./model2"

st.set_page_config(page_title="Next Word Predictor", layout="centered")



@st.cache_resource
def load_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    word2idx = checkpoint["word2idx"]
    idx2word = checkpoint["idx2word"]
    model = NextTokenPredictor(vocab_size=len(word2idx),emb_size=checkpoint["embed_dim"],hidden_size=checkpoint["hidden_dim"],block_size=checkpoint["context_length"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, word2idx, idx2word, checkpoint["context_length"]


#### Sidebar controls

st.sidebar.header("Decoding Settings")

context_length = st.sidebar.selectbox("Context length",[5, 10])

embed_dim = st.sidebar.selectbox("Embeding dimention",[32, 64])

act_fun = st.sidebar.selectbox("Activation function",['relu', 'tanh'])

max_num_words = st.sidebar.slider("Max new tokens", 1, 20, 10)

temperature = st.sidebar.slider("Creativity (Temperature)", 0.2, 2.0, 1.0)

top_k = st.sidebar.slider("Top-k", 0, 20, 10)


#### Main UI

st.title("Next-Word Prediction using MLP")
st.write("Generate text using an MLP-based natural language model")

#### Model selector

model_filename = f"model_cont{str(context_length)}_emb{embed_dim}_{act_fun}.pt"
model_path = os.path.join(MODELS_DIR, model_filename)

if not os.path.exists(model_path):
    st.error(f"Model not found: {model_filename}\nPlease ensure this file exists in ./Models/")
    st.stop()

# Load selected model
model, word2idx, idx2word, context_length = load_model(model_path)

# User input
user_input = st.text_input("Enter your starting text:",autocomplete="off")

if st.button("Generate"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        result = generate_next_words(model, word2idx, idx2word, user_input, act_fun,max_num_words, temperature, context_length,top_k)
        st.subheader("Generated Text")
        st.write(result)

st.markdown("---")