import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ====================
# Load Saved Models and Tokenizers
# ====================
@st.cache_resource
def load_all():
    with open('eng_tokenizer.pkl', 'rb') as f:
        eng_tokenizer = pickle.load(f)
    with open('hin_tokenizer.pkl', 'rb') as f:
        hin_tokenizer = pickle.load(f)

    encoder_model = load_model('encoder_model.h5', compile=False)
    decoder_model = load_model('decoder_model.h5', compile=False)

    with open("seq_lengths.pkl", "rb") as f:
        max_eng_len, max_hin_len = pickle.load(f)

    return eng_tokenizer, hin_tokenizer, encoder_model, decoder_model,max_eng_len, max_hin_len

eng_tokenizer, hin_tokenizer, encoder_model, decoder_model,max_eng_len, max_hin_len = load_all()

# ====================
# Reverse Hindi Token Index
# ====================
reverse_hin_index = {i: w for w, i in hin_tokenizer.word_index.items()}

# ====================
# Translation Function
# ====================
reverse_hin_index = {i: w for w, i in hin_tokenizer.word_index.items()}

def translate(sentence):
    seq = eng_tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=max_eng_len, padding='post')
    enc_h, enc_c = encoder_model.predict(seq)
    states = [enc_h, enc_c]

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = hin_tokenizer.word_index["start"]

    stop = False
    translated = []

    while not stop:
        output_tokens, h, c = decoder_model.predict([target_seq] + states)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_hin_index.get(sampled_token_index, '')

        if sampled_word == "end" or len(translated) > max_hin_len:
            stop = True
        elif sampled_word != "start":
            translated.append(sampled_word)

        target_seq[0, 0] = sampled_token_index
        states = [h, c]

    return ' '.join(translated)

# ====================
# Streamlit UI
# ====================
st.title("English to Hindi Translator")
st.write("Enter an English sentence below:")

user_input = st.text_input("English Sentence", "how are you")

if st.button("Translate"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        hindi_output = translate(user_input)
        st.success("**Hindi Translation:** " + hindi_output)
