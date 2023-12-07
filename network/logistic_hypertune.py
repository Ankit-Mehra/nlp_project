"""
hyperparameter tuning for logistic regression
"""
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def hypertune_logistic(x_train:pd.Series,
                        y_train:pd.Series)-> GridSearchCV:
    """
    Perform hyperparameter tuning for logistic regression
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('logistic', LogisticRegression(max_iter=3000))
    ])

    param_grid = {
    'logistic__C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
    'logistic__penalty': ['l2'],              # Penalty type
    'logistic__solver': ['liblinear', 'lbfgs', "sag"]# Solver algorithm
    }

    grid_search = GridSearchCV(pipeline,
                                 param_grid,
                                 cv=5,
                                 n_jobs=-1,
                                 verbose=1)

    grid_search.fit(x_train, y_train)

    return grid_search
    