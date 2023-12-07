"""
Script to perform under sampling on the dataset
"""
import pandas as pd

# Assuming df is your DataFrame and 'sentiment' is the class column
# And 'Positive', 'Neutral', 'Negative' are the class labels
def under_sample(df:pd.DataFrame)-> pd.DataFrame:
    """
    Perform under sampling on the dataset
    """
    # Split the DataFrame by class
    df_positive = df[df['sentiment'] == 'Positive']
    df_neutral = df[df['sentiment'] == 'Neutral']
    df_negative = df[df['sentiment'] == 'Negative']

    # Get the minority class
    minority_class = check_minority(df_positive,df_neutral,df_negative)

    sample_size = len(df[df['sentiment'] == minority_class])

    # Under sample the majority class
    df_postive_under = df_positive.sample(sample_size)
    df_negative_under = df_negative.sample(sample_size)
    df_neutral_under= df_neutral.sample(sample_size)

    df_balanced = pd.concat([df_postive_under,
                             df_neutral_under,
                             df_negative_under], axis=0)

    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)

    return df_balanced

def check_minority(df_positive,df_neutral,df_negative)-> str:
    """
    Check the minority class
    """
    # check the minority class
    if len(df_positive) < len(df_neutral) and len(df_positive) < len(df_negative):
        return "Positive"
    if len(df_neutral) < len(df_positive) and len(df_neutral) < len(df_negative):
        return "Neutral"
    return "Negative"
