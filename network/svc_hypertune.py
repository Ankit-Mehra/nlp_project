"""
Script to hypertune SVM model

"""
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pandas as pd

def svc_hypertune(x_train:pd.Series,
                    y_train:pd.Series)-> GridSearchCV:
    """
    Hypertune the SVC model
    """
    # create a pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svm', SVC(kernel='linear'))
    ])

    # define hyperparameters and ranges
    parameters = {
        'tfidf__max_features': [1000,2000,5000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'svm__C': [0.1,1, 10],
        'svm__kernel': ['linear', 'rbf'],
    }

    # hypertune the model
    print("Hypertuning the SVC model...")
    grid_search = GridSearchCV(pipeline,
                                parameters,
                                cv=5,
                                n_jobs=-1,
                                verbose=1)

    grid_search.fit(x_train, y_train)

    return grid_search
