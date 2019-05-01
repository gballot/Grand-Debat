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

def fill_X(auth_index):
    global gmm
    global ids_auth
    global features
    global four_surveys_taken_auth_ids
    auth = four_surveys_taken_auth_ids[auth_index]
    k = list(ids_auth).index(auth)
    return gmm.predict_proba(features[k].reshape(1, -1))[0]


n_compo = 10
df_ecologie = ut.read_data('data/LA_TRANSITION_ECOLOGIQUE.json')
df_resp_eco = get_open_reponses(df_ecologie)
df_ids_eco = get_ids_open_reponses(df_ecologie)
four_surveys_taken_auth_ids = np.loadtxt("four_surveys_taken_auth_ids.csv", delimiter=",", dtype=str)
ids_auth = np.sort(list(set(df_resp_eco['authorId'].values)))
np.savetxt("ids_auth_sorted.csv", ids_auth, delimiter=",", fmt="%s")
X = np.zeros((len(four_surveys_taken_auth_ids), n_compo))
# read features
features = np.loadtxt('responses ecologie_all_questions.tsv', delimiter='\t')
# Fit GMM
gmm = GaussianMixture(n_components=n_compo)
gmm.fit(features)
# pool
local_pool = multiprocessing.Pool(10)
X = np.array(local_pool.map(fill_X, range(len(four_surveys_taken_auth_ids))))
local_pool.close()
local_pool.join()
np.savetxt("X_ecologie.csv", X, delimiter=",")








