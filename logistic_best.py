"""
Script to train the best logistic regression model for sentiment analysis
and make predictions on the sample data
"""
import pandas as pd
import joblib
from preprocessing.pre_process import clean_data

def predict_sample():
    """
    predict sentiment on sample data
    """

    data = pd.read_csv('data/df_select.csv')

    #clean the data
    data = clean_data(data)

    #feature
    feature = data['clean_text']

    # load the best model from the pickle file
    model_path = 'models/log_best.pkl'
    label_encoder_path = 'models/log_label_encoder.pkl'

    best_model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)

    # make the predictions
    y_pred = best_model.predict(feature)

    # change numeric labels to string
    # labels of positive, negative and neutral using label encoder
    y_pred = label_encoder.inverse_transform(y_pred)

    data['logistic_predicted'] = y_pred

    # save the data
    data.to_csv('data/logistic_predicted.csv', index=False)

if __name__ == "__main__":
    predict_sample()
 