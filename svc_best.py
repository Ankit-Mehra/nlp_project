"""
Script to train the best svc model for sentiment analysis
"""
import pandas as pd
import joblib
from network.svc import best_param_svc
from preprocessing.pre_process import clean_data
from sample.under_sampling import under_sample

def train_best():
    """
    Main function
    """
    # read the data
    data = pd.read_csv('data/feature_target_data.csv')

    #class distribution
    print(data['sentiment'].value_counts()*100/len(data))

    # under sample the data
    data = under_sample(data)

    #class distribution
    print(data['sentiment'].value_counts()*100/len(data))

    # run the best model
    accuracy_train, accuracy_test = best_param_svc(data, 'sentiment')

    print("Accuracy score on train data: ", accuracy_train)
    print("Accuracy score on test data: ", accuracy_test)

def make_prediction_sample():
    """
    Make prediction on the sample data
    """
    # read the data
    data = pd.read_csv('data/df_select.csv')

    # clean the data
    data = clean_data(data)

    # feature and target data
    feature = data['clean_text']

    # load the best model from the pickle file
    model_path = 'models/svc_best.pkl'
    label_encoder_path = 'models/svc_label_encoder.pkl'

    best_model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)

    # make the predictions
    y_pred = best_model.predict(feature)

    # change numeric labels to string 
    # labels of positive, negative and neutral using label encoder
    y_pred = label_encoder.inverse_transform(y_pred)

    data['svc_predicted'] = y_pred

    # save the data
    data.to_csv('data/svc_predicted.csv', index=False)

if __name__ == "__main__":
    # train_best()
    make_prediction_sample()
