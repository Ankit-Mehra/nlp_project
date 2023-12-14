"""
Streamlit app to show the prediction of different models
"""
import time
import streamlit as st
from svc_best import make_prediction, get_random_text
from logistic_best import predict
from lexicon.textblob_lexicon import label_sentiment_textblob
from lexicon.vader_lexicon import label_sentiment_vader
from state_of_art import predict_helpful

# Initialize session state variables if they don't exist
if 'random_text' not in st.session_state:
    st.session_state['random_text'] = None
if 'label' not in st.session_state:
    st.session_state['label'] = None

st.title("Predicting sentiment(😊😔😐) of different models")

# button to get the random review
if st.button("Get random review"):
    random_text,random_help,label = get_random_text()
    st.write(random_text)

    # write label in bigger font
    st.markdown(f"<h3 style='text-align: center; color: white;'>Actual Label :{label}</h3>",
                unsafe_allow_html=True)

    # make the prediction
    with st.spinner("Predicting..."):
        time.sleep(3)
        with st.expander("Prediction from SVC"):
            prediction_svc = make_prediction(random_text)
            st.write(prediction_svc)
        with st.expander("Prediction from Logistic Regression"):
            prediction_log = predict(random_text)
            st.write(prediction_log)
        with st.expander("Prediction from TextBlob"):
            prediction_blob = label_sentiment_textblob(random_text)
            st.write(prediction_blob)
        with st.expander("Prediction from Vader"):
            prediction_vader = label_sentiment_vader(random_text)
            st.write(prediction_vader)
        with st.expander("Prediction with Helpfulness"):
            prediction_ensemble = predict_helpful(random_text,random_help)
            st.write(prediction_ensemble)
