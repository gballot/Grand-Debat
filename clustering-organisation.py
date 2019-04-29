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
    global df_organisation
    global df_resp_org
    global df_ids_org
    global X
    global ids_auth
    global n_compo
    n_compo = 10
    four_surveys_taken_auth_ids = np.loadtxt("four_surveys_taken_auth_ids.csv", delimiter=",", dtype=str)
    ids_auth = np.sort(list(set(df_resp_fis['authorId'].values)))
    X = np.zeros((len(ids_auth), n_compo))
    df_organisation = ut.read_data('data/ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS.json')
    df_resp_org = get_open_reponses(df_organisation)
    df_ids_org = get_ids_open_reponses(df_organisation)
    # read features
    features = np.loadtxt('response organisation_all_questions.tsv', delimiter='\t')
    # Fit GMM
    gmm = GaussianMixture(n_components=n_compo)
    gmm.fit(np.array(features))
    # pool
    local_pool = multiprocessing.Pool(20, initializer)
    local_pool.map(fill_X, range(four_surveys_taken_auth_ids))
    local_pool.close()
    local_pool.join()
    np.savetxt("X_organisation.csv", X, delimiter=",")


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






