"""
Script to vectorize the text data
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

def vectorize_tfidf(data_train:pd.Series,
                    data_test:pd.Series,
                    max_features:int=2000)-> (pd.Series, pd.Series):
    """
    Vectorize the text data using TF-IDF vectorizer
    """
    tfidf = TfidfVectorizer(max_features=max_features)

    print("Vectorizing the text data using TF-IDF...")
    x_train = tfidf.fit_transform(data_train)
    x_test = tfidf.transform(data_test)

    return x_train, x_test

def embed_text(data:pd.Series)-> pd.Series:
    """
    Embed the text data using Word2Vec
    """

    # tokenize the text
    tokenized_text = data.apply(simple_preprocess)

    # create the word2vec model
    w2v_model = Word2Vec(tokenized_text,
                         vector_size=100,
                         window=5, min_count=1)

    print("Embedding the text data using Word2Vec...")

    # convert documents to vectors by averaging the word vectors
    embedded_vector = tokenized_text.apply(lambda x:pd.Series(w2v_model.wv[x].mean(axis=0)))

    return embedded_vector
