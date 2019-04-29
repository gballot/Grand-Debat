#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 17:27:13 2019

@author: maxime
"""
#%%
import src.utils as ut
import numpy as np
import pandas as pd
import string
import multiprocessing
#from src.kmeans_embeddings import FeaturesExtractor
from src.utils import (read_data, get_open_reponses, get_ids_open_reponses)
from sklearn.mixture import GaussianMixture

def initializer():
    global gmm
    global features
    global df_democratie
    global df_resp_fis
    global df_ids_fis
    global X
    global ids_auth
    global n_compo
    n_compo = 10
    X = np.zeros((len(four_surveys_taken_auth_ids), n_compo))
    df_fiscalite = ut.read_data('data/LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES.json')
    df_resp_fis = get_open_reponses(df_fiscalite)
    df_ids_fis = get_ids_open_reponses(df_fiscalite)
    four_surveys_taken_auth_ids = np.loadtxt("four_surveys_taken_auth_ids.csv", delimiter=",", dtype=str)
    ids_auth = np.sort(list(set(df_resp_fis['authorId'].values)))
    # read features
    features = np.loadtxt('response fiscalite_all_questions.tsv', delimiter='\t')
    # Fit GMM
    gmm = GaussianMixture(n_components=n_compo)
    gmm.fit(np.array(features))
    # pool
    local_pool = multiprocessing.Pool(20, initializer)
    local_pool.map(fill_X, range(four_surveys_taken_auth_ids))
    local_pool.close()
    local_pool.join()
    np.savetxt("X_fiscalite.csv", X, delimiter=",")


def fill_X(auth_index):
    local_features = []
    k = ids_auth.index(auth)
    local_features = gmm.predict_proba(features[k].reshape(1, -1).ravel())
    X[auth_index] = local_features
    print(X)


if __name__ == '__main__':
    pool = multiprocessing.Pool(4)
    pool.map(features_from, range(4))
    pool.close()
    pool.join()
    np.savetxt("X.csv", X, delimiter=",")






