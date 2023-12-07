"""
This is the lexicon-based sentiment analysis tool, 
VADER (Valence Aware Dictionary and sEntiment Reasoner).
"""

import pandas as pd
from lexicon.vader_lexicon import predict_sentiment
from preprocessing.pre_process import clean_data

def main():
    """
    Main function
    """
    # read the data
    data = pd.read_csv('data/df_select.csv')

    # clean the data
    data = clean_data(data)
    
    #class distribution
    print(data['sentiment'].value_counts()*100/len(data))

    # predict the sentiment
    data = predict_sentiment(data)

    # save the data
    data.to_csv('data/vader_predicted.csv', index=False)

if __name__ == "__main__":
    main()
