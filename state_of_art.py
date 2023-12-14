"""
Script to run the state of the art models
"""
import pandas as pd
import joblib
import numpy as np
from scipy.sparse import hstack
from preprocessing.pre_process import clean_data_helpfulness

def predict_helpful_sample():
    """
    predict sentiment on sample data
    """

    data = pd.read_csv('data/df_select.csv')

    #clean the data
    data = clean_data_helpfulness(data)

    # Extract helpful votes and total votes from the 'helpful' column
    data['helpful_votes'] = data['helpful'].apply(lambda x: x[0])
    data['total_votes'] = data['helpful'].apply(lambda x: x[1])

    # Calculate the quality score for each review
    # Quality score = Number of helpful votes / Total number of votes
    # For reviews with no votes, the quality score will be NaN or 0 (we will handle this later)
    data['helpfullness'] = data['helpful_votes'] / data['total_votes']
    data['helpfullness'] = data['helpfullness'].fillna(0)
    
    #feature
    feature = data[['clean_text', 'helpfullness']]

    # load the best model from the pickle file
    model_path = 'models/svc_helpful.pkl'
    # label_encoder_path = 'models/log_label_encoder.pkl'

    best_model = joblib.load(model_path)
    # label_encoder = joblib.load(label_encoder_path)

    # make the predictions
    y_pred = best_model.predict(feature)

    # change numeric labels to string
    # labels of positive, negative and neutral using label encoder
    # y_pred = label_encoder.inverse_transform(y_pred)

    data['logistic_predicted'] = y_pred

    # save the data
    data.to_csv('data/state_svc_predicted.csv', index=False)
    
def predict_helpful(text,helpful):
    """
    predict sentiment on the given row
    """
    # load the best model from the pickle file
    model_path = 'models/svc_helpful.pkl'
    label_encoder_path = 'models/label_encoder_helpful.pkl'
    vectorizer_path = 'models/vectorizer_helpful.pkl'

    # load models
    best_model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    vectorizer = joblib.load(vectorizer_path)

    # change the helpful votes to list
    helpful = eval(helpful)

    if helpful[1] == 0:
        helpful_score = [0]
    else:
        helpful_score = int(helpful[0])/int(helpful[1])

    # reshape helpful_score to a 2D array
    helpful_score = np.array(helpful_score).reshape(1, -1)

    # vecotrize the text
    text = vectorizer.transform([text])

    # concatenate text and helpful_score
    features = hstack([text, helpful_score])

    y_pred = best_model.predict(features)

    # change numeric labels to string
    # labels of positive, negative and neutral using label encoder
    y_pred = label_encoder.inverse_transform(y_pred)

    return y_pred[0]

if __name__ == "__main__":
    predict_helpful_sample()
 