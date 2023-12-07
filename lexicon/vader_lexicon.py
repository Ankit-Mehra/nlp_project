"""
Script to do lexical analysis using VADER
"""
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def label_sentiment_vader(text):
    """
    Analyze the sentiment of the reviews
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    if sentiment['compound'] > 0.1:
        return 'Positive'
    if sentiment['compound'] < -0.1:
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

    print("Predicting the sentiment using VADER...")
    # predict the sentiment
    data['predicted_vader'] = data['clean_text'].apply(label_sentiment_vader)

    return data
