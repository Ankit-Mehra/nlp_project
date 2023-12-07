"""
Main file to svc model 
"""
import os
import pandas as pd
from network.svc import (svc_tfidf, svc_embedding,
                         hypertune_svc,best_param_svc,
                         best_svc_fit)
from preprocessing.pre_process import read_clean_data, clean_data
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

accuracy_score_train_tfidf, accuracy_score_test_tfidf = svc_tfidf(
    data_balanced, 'sentiment')

print("Accuracy score on train data with TFIDF: ", accuracy_score_train_tfidf)
print("Accuracy score on test data with TFIDF: ", accuracy_score_test_tfidf)

accuracy_score_train_embed, accuracy_score_test_embed = svc_embedding(
    data_balanced, 'sentiment')

print("Accuracy score on train data with word2vec embeding: ",
      accuracy_score_train_embed)
print("Accuracy score on test data with word2vec embeding: ",
      accuracy_score_test_embed)

# hypertune the model
grid_train_accuracy, grid_test_accuracy = hypertune_svc(data_balanced,'sentiment')

print("Accuracy score on train data after hypertuning: ", grid_train_accuracy)
print("Accuracy score on test data after hypertuning: ", grid_test_accuracy)

# {'svm__C': 1,
#  'svm__kernel': 'rbf',
#  'tfidf__max_features': 5000,
#  'tfidf__ngram_range': (1, 2)}

# Accuracy score on train data after hypertuning:  0.6434502515633773
# Accuracy score on test data after hypertuning:  0.6561757719714965


    