#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 17:27:13 2019

@author: gabriel
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

df_responses = get_open_reponses(df_fiscalite)

responses = (df_responses[:].formattedValue.values.tolist())

# Extract embeddings for sentences
s = FeaturesExtractor()
features = [s.get_features(x) for x in responses]

features_np = np.array(features)

#samples_id = np.random.choice(range(len(features)), 5000)

features_np_samples = features_np[:,:]#samples_id, :]
np.savetxt('features_s_fiscalite_all_questions.tsv', features_np_samples, delimiter='\t')
#responses_samples = [responses[i] for i in samples_id]
with open('labels_s_fiscalite_all_questions.tsv', 'w') as f:
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
