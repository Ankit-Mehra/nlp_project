"""
Script to perform SMOTE on the dataset

"""
import pandas as pd
from imblearn.over_sampling import SMOTE

def read_data(path:str)-> pd.DataFrame:
    """Reads the data from the csv file"""

    data = pd.read_csv(path)

    return data

def save_data(data:pd.DataFrame, path:str):
    """Saves the data to the csv file"""

    data.to_csv(path, index=False)


def perform_smote(feature:pd.Series, target:pd.Series, k_neighbors:int=5, random_state:int=42):
    """Performs SMOTE on the dataset"""

    sm = SMOTE(k_neighbors=k_neighbors, random_state=random_state)

    x_smote, y_smote = sm.fit_resample(feature, target)

    return x_smote, y_smote



# def main():
#     """Main function"""

#     data = read_data('data/office_products_cleaned.csv')

#     X_sm, y_sm = perform_smote(data, 'target')

#     data_sm = pd.concat([X_sm, y_sm], axis=1)

#     save_data(data_sm, 'data/office_products_smote.csv')