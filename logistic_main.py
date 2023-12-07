"""
Script to run the logistic regression model
"""
import os
import pandas as pd
from network.logistic import logistic_tfidf, logistic_embedding, hypertune_log
from preprocessing.pre_process import read_clean_data
from sample.under_sampling import under_sample

# get the data
DATA_PATH  = "data/reviews_Office_Products_5.json.gz"
SAMPLE_DATA = "data/df_select.csv"
FEATUE_TARGET_DATA = "feature_target_data.csv"

if FEATUE_TARGET_DATA not in os.listdir('data'):
    data = read_clean_data(data_path=DATA_PATH,
                                   sample_path=SAMPLE_DATA)
    print("Data loaded from the json.gz file.")
else:
    path = os.path.join('data', FEATUE_TARGET_DATA)
    data = pd.read_csv(path)
    print("Data loaded from the csv file.")

# value counts of the target variable before under sampling
print(data['sentiment'].value_counts()*100/len(data))

# perform under sampling
data_balanced = under_sample(data)

# value counts of the target variable after under sampling
print(data_balanced['sentiment'].value_counts()*100/len(data_balanced))

# accuracy_score_train_tfidf, accuracy_score_test_tfidf = logistic_tfidf(
#     data_balanced, 'sentiment')

# print("Accuracy score on train data with TFIDF: ", accuracy_score_train_tfidf)
# print("Accuracy score on test data with TFIDF: ", accuracy_score_test_tfidf)

# # Accuracy score on train data with TFIDF:  0.7895451459606245
# # Accuracy score on test data with TFIDF:  0.6357878068091845

# accuracy_score_train_embed, accuracy_score_test_embed = logistic_embedding(
#     data_balanced, 'sentiment')

# print("Accuracy score on train data with word2vec embeding: ",
#         accuracy_score_train_embed)
# print("Accuracy score on test data with word2vec embeding: ",
#         accuracy_score_test_embed)

# Accuracy score on train data with word2vec embeding:  0.5648336727766463
# Accuracy score on test data with word2vec embeding:  0.35589865399841647

accuracy_train_hypertune,accuracy_test_hypertune = hypertune_log(
    data_balanced, 'sentiment')

print("Accuracy score on train data with hyperparameter tuning: ",
        accuracy_train_hypertune)
print("Accuracy score on test data with hyperparameter tuning: ",
        accuracy_test_hypertune)
