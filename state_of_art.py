"""
Script to run the state of the art models
"""
import pandas as pd
import joblib
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

if __name__ == "__main__":
    predict_helpful_sample()
 