# -*- coding: utf-8 -*-

"""
Simple Kmeans clustering with TF-IDF features
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import stop_words
from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
import pandas as pd


def get_stop_words():
    custom_stop_words = set(stopwords.words('french') +
                            list(string.punctuation) +
                            stop_words.get_stop_words('fr'))
    return custom_stop_words


def tokenize(text):
    return word_tokenize(text, language='french')


def kmeans_tfidf(df_responses: pd.DataFrame,
                 n_clusters: int = 8,
                 max_features: int = 3000,
                 questionId: str = '107'):
    """
    """
    responses = df_responses[df_responses.questionId == '107']
    answers = responses.formattedValue.str.lower().values.tolist()

    custom_stop_words = get_stop_words()
    vectorizer = TfidfVectorizer(stop_words=custom_stop_words,
                                 tokenizer=tokenize,
                                 max_features=max_features)

    X = vectorizer.fit_transform(answers)
    words = vectorizer.get_feature_names()

    # elbow methods
    # wcss = []
    # for i in range(1, 11):
    #     kmeans = KMeans(n_clusters=i, init='k-means++',
    #                     max_iter=300, n_init=10, random_state=0)
    #     kmeans.fit(X3)
    #     wcss.append(kmeans.inertia_)
    # plt.plot(range(1, 11), wcss)
    # plt.title('The Elbow Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('WCSS')
    # # plt.savefig('elbow.png')
    # plt.show()

    kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=-1)
    kmeans.fit(X)

    # Finally, we look at the clusters generated by k-means.
    common_words = kmeans.cluster_centers_.argsort()[:, -1:-26:-1]
    for num, centroid in enumerate(common_words):
        print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))

    return kmeans, common_words
