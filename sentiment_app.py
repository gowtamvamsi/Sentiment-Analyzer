import streamlit as st
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle

st.title('Sentiment analyzer of Reviews')
default_value_goes_here = "The product is good, but I hated it"
user_input = st.text_input("Enter your review here", default_value_goes_here)

model = load_model('model.h5')
tokenizer = ''
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_sentiment_of_the_review(review):
    encoded_doc_test = tokenizer.texts_to_sequences([review])
    padded_text_test=pad_sequences(encoded_doc_test,maxlen=4000, padding="post", truncating="post").astype('int16')
    res = model.predict(padded_text_test)
    print(res[0][0])
    if res[0][0]<0.5:
        st.write("It's a negative review")
        ans = False
    else:
        st.write("It's a positive review")
        ans = True
    return ans
if st.button('Predict the sentiment'):
    if user_input:
        predict_sentiment_of_the_review(user_input)
