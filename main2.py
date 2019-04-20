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

#%%

questionId = '162'

df_responses = get_open_reponses(df_fiscalite)

responses = (df_responses[df_responses.questionId == questionId].formattedValue.values.tolist())

features = np.loadtxt('features_s_fiscalite_'+questionId+'.tsv', delimiter='\t')
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
