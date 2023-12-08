"""
Module to train the SVC model using TF-IDF vectorizer and SMOTE
"""
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from preprocessing.vectorizer import vectorize_tfidf, embed_text
from preprocessing.pre_process import split_stratfied
from network.svc_hypertune import svc_hypertune

def svc_fit(feature:pd.Series, target:pd.Series)-> SVC:
    """
    Train the SVC model using TF-IDF vectorizer
    """
    # train the SVC model
    svc = SVC()
    svc.fit(feature, target)

    return svc

def best_svc_fit(feature:pd.Series, target:pd.Series)-> SVC:
    """
    Train the SVC model with following parameters:
    {'svm__C': 10,
    #  'svm__kernel': 'rbf',
    #  'tfidf__max_features': 5000,
    #  'tfidf__ngram_range': (1, 2)}
    """
    # train the SVC model
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000,
                                  ngram_range=(1,2))),
        ('svm', SVC(C=10,kernel='rbf'))
    ])

    svc = pipeline.fit(feature, target)

    return svc

def svc_tfidf(df:pd.DataFrame,target_col:str)->(float,float):
    """
    Train the SVC model using SMOTE
    """
    # split the data into train and test sets
    x_train, y_train, x_test, y_test = split_data(df, target_col)

    # encode the target variables using label encoder
    y_train = LabelEncoder().fit_transform(y_train)
    y_test = LabelEncoder().fit_transform(y_test)

    # vectorize the train and test data
    x_train_vec,x_test_vec = vectorize_tfidf(x_train,x_test)

    # train the SVC model
    print("Training the SVC model...")
    svc = svc_fit(x_train_vec, y_train)

    # Predictions
    y_pred_train = svc.predict(x_train_vec)
    y_pred_test = svc.predict(x_test_vec)

    # Evaluate the model
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    return accuracy_train, accuracy_test


def svc_embedding(df:pd.DataFrame,target_col:str)-> (float,float):
    """
    Train the SVC model using Word2Vec
    """
    # split the data into train and test sets
    x_train, y_train, x_test, y_test = split_data(df, target_col)

    x_train_w2v = embed_text(x_train)
    x_test_w2v = embed_text(x_test)

    # train the SVC model
    print("Training the SVC model with word2vec...")
    svc = svc_fit(x_train_w2v, y_train)

    # Predictions
    y_pred_train = svc.predict(x_train_w2v)
    y_pred_test = svc.predict(x_test_w2v)

    # Evaluate the model
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    return accuracy_train, accuracy_test

def hypertune_svc(df:pd.DataFrame,target_col:str)-> (float,float):
    """
    Hypertune the SVC model
    """
    # split the data into train and test sets
    x_train, y_train, x_test, y_test = split_data(df, target_col)
    
    # encode the target variables using label encoder
    labler = LabelEncoder()
    y_train = labler.fit_transform(y_train)
    y_test = labler.fit_transform(y_test)

    print("Hypertuning the SVC model...")
    # hypertune the model
    grid_search = svc_hypertune(x_train, y_train)

    # best parameters
    print("Best parameters: ", grid_search.best_params_)

    # best score
    print("Best score: ", grid_search.best_score_)

    # evaluate the model on test data
    print("Accuracy score on test data: ", grid_search.score(x_test, y_test))

    return grid_search.best_score_, grid_search.score(x_test, y_test)

def split_data(df,target_col:str)->(pd.DataFrame,pd.DataFrame,
                                    pd.DataFrame,pd.DataFrame):
    """
    split the data into train and test sets
    """

    # split the data into train and test sets
    strat_train, strat_test = split_stratfied(df, target_col,
                                              test_size=0.3,
                                              random_state=42)

    # split the data into train and test sets
    x_train,y_train = strat_train['clean_text'], strat_train['sentiment']
    x_test,y_test = strat_test['clean_text'], strat_test['sentiment']

    return x_train, y_train, x_test, y_test


def best_param_svc(df:pd.DataFrame,target_col:str)-> SVC:
    """
    Train the SVC model using the best parameters
    """
    # split the data into train and test sets
    x_train, y_train, x_test, y_test = split_data(df, target_col)

    # encode the target variables using label encoder
    labler = LabelEncoder()
    y_train = labler.fit_transform(y_train)
    y_test = labler.fit_transform(y_test)

    # train the best SVC model
    print("Training the best SVC model...")
    svc = best_svc_fit(x_train, y_train)

    # save the label encoder
    joblib.dump(labler, './models/svc_label_encoder.pkl')

    # save the model
    joblib.dump(svc, './models/svc_best.pkl')

    # predictions
    y_pred_train = svc.predict(x_train)
    y_pred_test = svc.predict(x_test)

    # Evaluate the model
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    return accuracy_train, accuracy_test
