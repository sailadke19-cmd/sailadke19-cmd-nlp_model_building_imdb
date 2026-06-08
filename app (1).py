import streamlit as st
import pickle

# Load the model and vectorizer
# Assuming these files are in the same directory as app.py
model = pickle.load(open("sentiment_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

st.title("IMDB Sentiment Analysis")

review = st.text_area("Enter Review")

if st.button("Predict"):

    review_vector = vectorizer.transform(
        [review]
    )

    prediction = model.predict(
        review_vector
    )

    st.write(
        "Prediction:",
        prediction[0]
    )
