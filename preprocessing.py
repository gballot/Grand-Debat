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
from src.utils import (read_data, get_open_reponses)
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
df_resp_dem = get_open_reponses(df_democratie)
df_resp_eco = get_open_reponses(df_ecologie)
df_resp_org = get_open_reponses(df_organisation)

dfs_responses = np.array([["responses fiscalite", df_resp_fis], ["responses democratie", df_resp_dem], ["responses ecologie", df_resp_eco], ["responses organisation", df_resp_org]])

#%% Be careful, these cells take a lot of time to run !!!!!
    
# allAuthIds is the sets of all the authorIds
allAuthIds = []
for i in range(4):
    allAuthIds.extend(set(dfs[i,1]['authorId'].values))
allAuthIds = set(allAuthIds)

# all_auth_id_array is the sorted array of all the authorIds
all_auth_id_array = np.sort(np.array(list(allAuthIds)))

# auth_answers_count[i,j] is 1 if all_auth_id_array[i] has answered survey dfs[j]
auth_answers_count = np.zeros((len(allAuthIds), 4), dtype=int)
for j in range(4):
    for i in range(len(all_auth_id_array)):
        auth = all_auth_id_array[i]
        line = dfs[j,1].loc[dfs[j,1]['authorId'] == auth]
        if(len(line) > 0):
            auth_answers_count[i,j] = auth_answers_count[i,j] + 1

            
# This cell aims to save the auth_answers_count array in auth_answers_count.csv
np.savetxt("auth_answers_count.csv", auth_answers_count, fmt='%1u', delimiter=",")

#%% extract features
for i in range(4):

    responses = (dfs_responses[i,1][:].formattedValue.values.tolist())

    # Extract embeddings for sentences
    s = FeaturesExtractor()
    features = [s.get_features(x) for x in responses]

    features_np = np.array(features)

    #samples_id = np.random.choice(range(len(features)), 5000)

    features_np_samples = features_np[:,:]#samples_id, :]
    np.savetxt(dfs_responses[i,0,]+'_all_questions.tsv', features_np_samples, delimiter='\t')
    #responses_samples = [responses[i] for i in samples_id]
    with open('labels_'+dfs_responses[i,0]+'_all_questions.tsv', 'w') as f:
        for resp in responses:#_samples:
            v = resp.replace('\n', '. ')
            v = v.replace('\t', '. ')
            f.write('{}\n'.format(v))
    # Fit GMM
    gmm = GaussianMixture(n_components=10)
    labels = gmm.fit_predict(np.array(features))

    # print samples from each clusters
    df = pd.DataFrame({'label': labels, 'response': responses})

    for label in df.label.unique():
        print('label {}'.format(label))
        samples = [x for x in df[df.label == label].sample(10).response.tolist()]
        for sample in samples:
             print(sample)
             print('#'*20)
