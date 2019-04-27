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
from src.kmeans_embeddings import FeaturesExtractor
from src.utils import (read_data, get_open_reponses, get_ids_open_reponses)
from sklearn.mixture import GaussianMixture

#%% extract data from json
df_fiscalite = ut.read_data('data/LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES.json')
df_democratie = ut.read_data('data/DEMOCRATIE_ET_CITOYENNETE.json')
df_ecologie = ut.read_data('data/LA_TRANSITION_ECOLOGIQUE.json')
df_organisation = ut.read_data('data/ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS.json')

dfs = np.array([["fiscalite", df_fiscalite], ["democratie", df_democratie], ["ecologie", df_ecologie], ["organisation", df_organisation]])
#%%

#%% responses of each themes
df_resp_fis = get_open_reponses(df_fiscalite)
df_ids_fis = get_ids_open_reponses(df_fiscalite)
df_resp_dem = get_open_reponses(df_democratie)
df_ids_dem = get_ids_open_reponses(df_democratie)
df_resp_eco = get_open_reponses(df_ecologie)
df_ids_eco = get_ids_open_reponses(df_ecologie)
df_resp_org = get_open_reponses(df_organisation)
df_ids_org = get_ids_open_reponses(df_organisation)

dfs_responses = np.array([["responses fiscalite", df_resp_fis], ["responses democratie", df_resp_dem], ["responses ecologie", df_resp_eco], ["responses organisation", df_resp_org]])
dfs_ids = np.array([df_ids_fis, df_ids_dem, df_ids_eco, df_ids_org])
#%% features and gmm for all themes

features_tab = []
gmms = []


for i in range(4):
    # read features
    features_tab[i] = np.loadtxt(dfs_responses[i,0,]+'_all_questions.tsv', delimiter='\t')

    # Fit GMM
    gmms[i] = GaussianMixture(n_components=10)
    labels = gmms[i].fit_predict(np.array(features_tab[i]))


#%% get answers to 4 themes

# auth_answers_count[i,j] is 1 if all_auth_id_array[i] has answered survey dfs[j]
# Read auth_answers_count from auth_answers_count.csv
auth_answers_count = np.loadtxt('auth_answers_count.csv', dtype=int ,delimiter=",")
# number_of_survey_taken[i] is the number of survey answered by all_auth_id_array[i]
number_of_survey_taken = np.sum(auth_answers_count, axis=1)
# number_of_participants_by_survey[i] is the number of participants to survey dfs[j]
number_of_participants_by_survey = np.sum(auth_answers_count, axis=0)

# allAuthIds is the sets of all the authorIds
allAuthIds = []
for i in range(4):
    allAuthIds.extend(set(dfs[i,1]['authorId'].values))
allAuthIds = set(allAuthIds)

# all_auth_id_array is the sorted array of all the authorIds
all_auth_id_array = np.sort(np.array(list(allAuthIds)))

four_surveys_taken_auth_ids = [all_auth_id_array[i] for i in range(len(all_auth_id_array)) if number_of_survey_taken[i] == 4]

print(four_surveys_taken_auth_ids)

#%% get 2nd clustering features

X = []

for auth in four_surveys_taken_auth_ids:
    features = []
    for i in range(4) :
        k = list(dfs_ids[i]).index(auth)
        features = np.concatenate((features,np.array(gmms[i].predict_proba(features_tab[i][k])).ravel()), axis=1)
    X.append(features)

print(X)



