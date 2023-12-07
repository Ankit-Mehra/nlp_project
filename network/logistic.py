"""
Script to implement logistic regression
"""
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from network.svc import split_data
from preprocessing.vectorizer import vectorize_tfidf, embed_text
from plots.plot import plot_confusion_matrix
from network.logistic_hypertune import hypertune_logistic

def logistic_regression(x_train_vec, y_train)-> LogisticRegression:
    """
    Logistic regression
    """
    model_logistic = LogisticRegression(max_iter=3000)

    # Train the model on the training set
    model_logistic.fit(x_train_vec, y_train)

    return model_logistic

def logistic_best(x_train_vec, y_train)-> LogisticRegression:
    """
    Logistic regression with best parameters
    """
    model_logistic = LogisticRegression(C=1,penalty='l2',
                                        solver='liblinear',
                                        max_iter=3000)

    # Train the model on the training set
    model_logistic.fit(x_train_vec, y_train)

    return model_logistic

def logistic_tfidf(data:pd.DataFrame,target_col:str)-> (float,float):
    """
    Logistic regression with TFIDF
    """
    # split the data into train and test sets
    x_train, y_train, x_test, y_test = split_data(data, target_col)

    # vectorize the data
    x_train_vec,x_test_vec = vectorize_tfidf(x_train,x_test)

    # train the model
    log_model = logistic_regression(x_train_vec, y_train)

    #predictions
    y_pred_train = log_model.predict(x_train_vec)
    y_pred_test = log_model.predict(x_test_vec)

    # acccuracy score
    training_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    # plot the confusion matrix
    plot_confusion_matrix(y_test, y_pred_test)

    return training_accuracy, test_accuracy

def logistic_embedding(data:pd.DataFrame,target_col:str)-> (float,float):
    """
    Logistic regression with Word2Vec
    """
    # split the data into train and test sets
    x_train, y_train, x_test, y_test = split_data(data, target_col)

    # vectorize the data
    x_train_w2vec = embed_text(x_train)
    x_test_w2vec = embed_text(x_test)

    # train the model
    log_model = logistic_regression(x_train_w2vec, y_train)

    #predictions
    y_pred_train = log_model.predict(x_train_w2vec)
    y_pred_test = log_model.predict(x_test_w2vec)

    # acccuracy score
    training_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    # plot the confusion matrix
    plot_confusion_matrix(y_test, y_pred_test)

    return training_accuracy, test_accuracy

def hypertune_log(data:pd.DataFrame,target_col:str)-> LogisticRegression:
    """
    Perform hyperparameter tuning for logistic regression
    """

   # split the data into train and test sets
    x_train, y_train, x_test, y_test = split_data(data, target_col)

    labeler = LabelEncoder()
    y_train = labeler.fit_transform(y_train)
    y_test = labeler.fit_transform(y_test)

    # hypertune the model
    grid_search = hypertune_logistic(x_train, y_train)

    # get the best model
    best_model = grid_search.best_estimator_

    # print the best parameters
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    print("Best model: ", best_model)

    #save the best model
    joblib.dump(best_model, './models/log_best.pkl')

    #save the label encoder
    joblib.dump(labeler, './models/log_label_encoder.pkl')

    #predictions
    y_pred_train = best_model.predict(x_train)
    y_pred_test = best_model.predict(x_test)

    # acccuracy score
    training_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    return training_accuracy, test_accuracy
