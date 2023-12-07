"""
Script to implement lexical analysis
"""

import pandas as pd
from textblob import TextBlob

def label_sentiment_textblob(text):
    """
    label the sentiment of the reviews
    """
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1:
        return 'Positive'
    if analysis.sentiment.polarity < -0.1:
        return 'Negative'
    return 'Neutral'

def predict_sentiment(data:pd.DataFrame)-> pd.DataFrame:
    """
    Predict the sentiment of the reviews
    """
    # drop the row of missing values
    data.dropna(subset=['clean_text'], inplace=True)

    # make the clean_text column as string
    data['clean_text'] = data['clean_text'].apply(str)
    print("Predicting the sentiment using TextBlob...")
    # predict the sentiment
    data['predicted_blob'] = data['clean_text'].apply(label_sentiment_textblob)

    return data
