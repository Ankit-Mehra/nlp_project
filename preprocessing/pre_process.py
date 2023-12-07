"""
Script to clean the data
"""
import re
import gzip
import json
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedShuffleSplit

def parse(path:str)-> dict:
    """
    Parse the json.gz file
    """
    with gzip.open(path, 'rb') as f:
        for line in f:
            yield json.loads(line)

def get_data(path:str)-> pd.DataFrame or None:
    """
    Get the dataframe from the json.gz file
    """
    try:
        df = dict(enumerate(parse(path)))
        return pd.DataFrame.from_dict(df, orient='index')
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def exclude_sample(df:pd.DataFrame, sample:pd.DataFrame)-> pd.DataFrame:
    """
    exclude the sample from the dataframe
    """
    sample_index = sample.index
    print(f"Excluding {len(sample_index)} samples from the dataframe.")
    return df.drop(sample_index)

def clean_text(text):
    """
    clean the text by removing non-alphabetic characters,
    converting to lowercase, removing stopwords and 
    stemming the remaining words
    """
    stop_words = set(stopwords.words('english'))
    stop_words = remove_negative(stop_words)

    # initialize the stemmer
    ps = PorterStemmer()

    # clean the text by removing non-alphabetic characters
    clean_text = re.sub('[^a-zA-Z]', ' ', text)

    # remove punctuations, numbers and special characters
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', clean_text)

    # convert the text to lowercase
    lowercase_text = clean_text.lower().split()

    # remove stopwords and stem the remaining words
    final_text = [ps.stem(word) for word in lowercase_text if word not in stop_words]
    return ' '.join(final_text)


def split_stratfied(df:pd.DataFrame,target_col:str,
                    test_size,random_state)-> (pd.DataFrame, pd.DataFrame):
    """
    Split the dataframe into train and test sets
    """

    split = StratifiedShuffleSplit(n_splits=1,
                                   test_size=test_size,
                                   random_state=random_state)

    print("Splitting the data into train and test sets...")

    for train_index, test_index in split.split(df, df[target_col]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]

    return strat_train_set, strat_test_set


def label(row)-> str:
    """
    label the data based on the rating
    """
    if row>=4:
        return "Positive"
    if row==3:
        return "Neutral"
    return "Negative"

def concat_columns(data, column1, column2)-> pd.Series:
    """
    concatenate two columns
    """
    return data[column1] + " " + data[column2]

def drop_na(data)-> pd.DataFrame:
    """
    drop columns with null values
    """
    data.dropna(inplace=True)

    # reset the index
    data.reset_index(drop=True, inplace=True)

    return data

def read_clean_data(data_path:str,
                    sample_path:str)-> pd.DataFrame:
    """
     Read the data, clean it and return the feature and target variables
    """
    data = get_data(data_path)

    # get the sample data
    sample_data = pd.read_csv(sample_path)

    # exclude the sample from the dataframe
    data_sample_excluded = exclude_sample(data, sample_data)

    # clean the data
    feature_target_data = clean_data(data_sample_excluded)

    # save the dataframe as a csv file
    feature_target_data.to_csv('data/feature_target_data.csv', index=False)

    return feature_target_data

def remove_negative(stop_words):
    """
    remove the negative words from the stopwords list
    """
    negetive_words = ['not','never','no','none',
                      'nothing','nowhere','neither',
                      'nor','nobody','hardly','scarcely',
                      'barely','doesnt','isnt','wasnt',
                      'shouldnt','wouldnt','couldnt',
                      'wont','cant','dont','arent','aint']

    # iterate through the stopwords list and remove the negative words from it
    # find common word in both the lists
    for word in negetive_words:
        if word in stop_words:
            stop_words.remove(word)
            
    return stop_words

def clean_data(data_sample_excluded)-> pd.DataFrame:
    """
    Clean the colums 'reviewText' and 'summary' and
    return the clean text
    """
    #concatenate the 'reviewText' and 'summary' columns
    data_sample_excluded['reveiwTextSummary'] = concat_columns(data_sample_excluded,
                                                               'reviewText', 'summary')

    #add a sentiment column with labels positive and negative and neutral
    data_sample_excluded['sentiment'] = data_sample_excluded['overall'].apply(label)

    #drop the null values and reset the index
    data_sample_excluded = drop_na(data_sample_excluded)

    print("Cleaning the text...")
    #clean the reviewTextSummary column
    data_sample_excluded['clean_text'] = data_sample_excluded['reveiwTextSummary'].apply(clean_text)

    x_train = data_sample_excluded['clean_text']
    y_train = data_sample_excluded['sentiment']

    # make a dataframe with the clean_text and sentiment columns
    feature_target_data = pd.DataFrame()
    feature_target_data['clean_text'] = x_train
    feature_target_data['sentiment'] = y_train

    return feature_target_data
