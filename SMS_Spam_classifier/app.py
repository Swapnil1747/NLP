import os
import pickle
import streamlit as st
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

def load_model_and_vectorizer():
    if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl'):
        try:
            model = pickle.load(open('model.pkl', 'rb'))
            vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
            check_is_fitted(model)
            return model, vectorizer
        except Exception as e:
            st.error(f"Error loading model or vectorizer: {e}")
            return None, None
    else:
        st.error("Model or vectorizer file not found. Please run retrain_model.py to generate them.")
        return None, None

model, tfidf = load_model_and_vectorizer()

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if model is None or tfidf is None:
        st.error("Model or vectorizer not loaded. Cannot perform prediction.")
    else:
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        try:
            result = model.predict(vector_input)[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            result = None

        # 4. Display
        if result is not None:
            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")
