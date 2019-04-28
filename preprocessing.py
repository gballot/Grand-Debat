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
df_resp_dem = get_open_reponses(df_democratie)
df_resp_eco = get_open_reponses(df_ecologie)
df_resp_org = get_open_reponses(df_organisation)

dfs_responses = np.array([["responses fiscalite", df_resp_fis], ["responses democratie", df_resp_dem], ["responses ecologie", df_resp_eco], ["responses organisation", df_resp_org]])
#%%

#%% extract features
s = FeaturesExtractor()
def extract_features():
    for k in range(4):

        ids_questions = get_ids_open_reponses(dfs[k,1])
        ids_auth = set(dfs_responses[k,1]['authorId'].values)

        responses = (dfs_responses[k,1][:].formattedValue.values.tolist())

        # Extract embeddings for sentences
        features = np.zeros((len(ids_auth), 300*len(ids_questions)))
        for i in range(len(ids_auth)) :
            for j in range(len(ids_questions)) :
                features[i][300*j:300*(j+1)] = s.get_features(dfs_responses[k,1][dfs_responses[k,1]['authorId','questionId'] == ids_auth[i],ids_questions[j]])



        np.savetxt(dfs_responses[k,0,]+'_all_questions.tsv', features, delimiter='\t')
        #responses_samples = [responses[i] for i in samples_id]
        #with open('labels_'+dfs_responses[k,0]+'_all_questions.tsv', 'w') as f:
        #    for resp in responses:#_samples:
        #        v = resp.replace('\n', '. ')
        #        v = v.replace('\t', '. ')
        #        f.write('{}\n'.format(v))

#%%

#extract_features()

