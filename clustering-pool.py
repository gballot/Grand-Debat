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

#%% extract data from json
#df_fiscalite = ut.read_data('data/LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES.json')
#df_democratie = ut.read_data('data/DEMOCRATIE_ET_CITOYENNETE.json')
#df_ecologie = ut.read_data('data/LA_TRANSITION_ECOLOGIQUE.json')
#df_organisation = ut.read_data('data/ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS.json')

#dfs = np.array([["fiscalite", df_fiscalite], ["democratie", df_democratie], ["ecologie", df_ecologie], ["organisation", df_organisation]])

four_surveys_taken_auth_ids = np.loadtxt("four_surveys_taken_auth_ids.csv", delimiter=",", dtype=str)
print(four_surveys_taken_auth_ids)

def initializer():
    global X
    X = np.zeros((len(four_surveys_taken_auth_ids), 4*10))

# pool function
def features_from(i):
    df_fiscalite, df_resp_fis, df_ids_fis, df_democratie, df_resp_dem, df_ids_dem, df_ecologie, df_resp_eco, df_ids_eco, df_organisation, df_resp_org, df_ids_org = 0,0,0,0,0,0,0,0,0,0,0,0
    gmm, features = 0,0
    print(df_fiscalite)
    if(i==0):
        df_fiscalite = ut.read_data('data/LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES.json')
        df_resp_fis = get_open_reponses(df_fiscalite)
        df_ids_fis = get_ids_open_reponses(df_fiscalite)
    elif(i==1):
        df_democratie = ut.read_data('data/DEMOCRATIE_ET_CITOYENNETE.json')
        df_resp_dem = get_open_reponses(df_democratie)
        df_ids_dem = get_ids_open_reponses(df_democratie)
    elif(i==2):
        df_ecologie = ut.read_data('data/LA_TRANSITION_ECOLOGIQUE.json')
        df_resp_eco = get_open_reponses(df_ecologie)
        df_ids_eco = get_ids_open_reponses(df_ecologie)
    elif(i==3):
        df_organisation = ut.read_data('data/ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS.json')
        df_resp_org = get_open_reponses(df_organisation)
        df_ids_org = get_ids_open_reponses(df_organisation)
    dfs = np.array([["fiscalite", df_fiscalite], ["democratie", df_democratie], ["ecologie", df_ecologie], ["organisation", df_organisation]])
    dfs_responses = np.array([["responses fiscalite", df_resp_fis], ["responses democratie", df_resp_dem], ["responses ecologie", df_resp_eco], ["responses organisation", df_resp_org]])
    dfs_ids = np.array([df_ids_fis, df_ids_dem, df_ids_eco, df_ids_org])
    # read features
    features = np.loadtxt(dfs_responses[i,0]+'_all_questions.tsv', delimiter='\t')
    # Fit GMM
    gmm = GaussianMixture(n_components=10)
    gmm.fit(np.array(features))
    local_pool = multiprocessing.Pool(20, initializer)
    local_pool.map(fill_X, range(four_surveys_taken_auth_ids))
    local_pool.close()
    local_pool.join()
    np.savetxt("X_"+str(i)+".csv", X, delimiter=",")

def fill_X(auth_index):
    local_features = []
    auth = four_surveys_taken_auth_ids[auth_index]
    ids_auth = np.sort(list(set(dfs_responses[k,1]['authorId'].values)))
    k = ids_auth.index(auth)
    local_features = gmm.predict_proba(features[k].reshape(1, -1).ravel())
    X[auth_index, 10*i:10*(i+1)] = local_features
    print(X)


if __name__ == '__main__':
    pool = multiprocessing.Pool(4)
    pool.map(features_from, range(4))
    pool.close()
    pool.join()
    np.savetxt("X.csv", X, delimiter=",")






