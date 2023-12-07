"""
Main file for textblob analysis
"""
import pandas as pd
from lexicon.textblob_lexicon import predict_sentiment
from preprocessing.pre_process import clean_data

def main():
    """
    Main function
    """
    # read the data
    data = pd.read_csv('data/df_select.csv')

    #clean the data
    data = clean_data(data)

    #class distribution
    print(data['sentiment'].value_counts()*100/len(data))

    # predict the sentiment
    data = predict_sentiment(data)

    # save the data
    data.to_csv('data/textblob_predicted.csv', index=False)


if __name__ == "__main__":
    main()
