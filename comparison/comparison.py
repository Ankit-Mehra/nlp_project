import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

def save_data(df,path):
    """
    Save data to csv file
    :return: None
    """
    df.to_csv(path, index=True, index_label='Algorithm')

def make_comparison_data()-> pd.DataFrame:
    """
    Make a dataframe for comparison
    :return: dataframe
    """
    texblob = pd.read_csv('./data/textblob_predicted.csv')
    vader = pd.read_csv('./data/vader_predicted.csv')
    svc_predicted_data = pd.read_csv('./data/svc_predicted.csv')
    logistic_predicted_data = pd.read_csv('./data/logistic_predicted.csv')

    df_comparison = pd.DataFrame()
    df_comparison['clean_text'] = texblob['clean_text']
    df_comparison['sentiment'] = texblob['sentiment']
    df_comparison['predicted_blob'] = texblob['predicted_blob']
    df_comparison['predicted_vader'] = vader['predicted_vader']
    df_comparison['predicted_svc'] = svc_predicted_data['svc_predicted']
    df_comparison['predicted_logistic'] = logistic_predicted_data['logistic_predicted']

    #save the dataframe
    save_data(df_comparison,'./data/comparison_data.csv')

    return df_comparison

def metrics_comparison_table():
    """
    Make a metrics comparison table
    :return: dataframe
    """
    metrics_table = pd.DataFrame(columns=['Accuracy','Precision','Recall','F1'],
                                 index=['Texblob','Vader','SVC','Logistic'])

    compare_data = pd.read_csv('./data/comparison_data.csv')

    #texblob
    metrics_table = add_metrics_to_comparison_table(compare_data,metrics_table,
                                                    'sentiment','predicted_blob',
                                                    'Texblob')

    #vader
    metrics_table = add_metrics_to_comparison_table(compare_data,metrics_table,
                                                    'sentiment','predicted_vader',
                                                    'Vader')

    #svc
    metrics_table = add_metrics_to_comparison_table(compare_data,metrics_table,
                                                    'sentiment','predicted_svc',
                                                    'SVC')

    #logistic
    metrics_table = add_metrics_to_comparison_table(compare_data,metrics_table,
                                                    'sentiment','predicted_logistic',
                                                    'Logistic')
    
    #save the dataframe
    save_data(metrics_table,'./data/metrics_comparison_table.csv')

    return metrics_table

def metrics_calculator(true_label,predicted_label):
    """
    Calculate the metrics
    """
    accuracy = accuracy_score(true_label,predicted_label)
    precision = precision_score(true_label,predicted_label,average='weighted')
    recall = recall_score(true_label,predicted_label,average='weighted')
    f1 = f1_score(true_label,predicted_label,average='weighted')

    return accuracy,precision,recall,f1

def add_metrics_to_comparison_table(df,metric_table,
                                    true_label,predicted_label,index):
    """
    add metrics to comparison table
    """
    true_label = df[true_label]
    predicted_label = df[predicted_label]
    accuracy,precision,recall,f1 = metrics_calculator(true_label,predicted_label)
    metric_table.loc[index] = [accuracy,precision,recall,f1]

    return metric_table
