#%% librairies imported
import src.utils as ut
import numpy as np
import pandas as pd
import string
#from src.kmeans_embeddings import FeaturesExtractor
from src.utils import (read_data, get_open_reponses)
from sklearn.cluster import KMeans

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

# allAuthIds is the sets of all the authorIds
allAuthIds = []
for i in range(4):
    allAuthIds.extend(set(dfs_responses[i,1]['authorId'].values))
allAuthIds = set(allAuthIds)

# all_auth_id_array is the sorted array of all the authorIds
all_auth_id_array = np.sort(np.array(list(allAuthIds)))

# auth_answers_count[i,j] is 1 if all_auth_id_array[i] has answered survey dfs[j]
auth_answers_count = np.zeros((len(allAuthIds), 4), dtype=int)
for j in range(4):
    for i in range(len(all_auth_id_array)):
        auth = all_auth_id_array[i]
        line = dfs_responses[j,1].loc[dfs_responses[j,1]['authorId'] == auth]
        if(len(line) > 0):
            auth_answers_count[i,j] = auth_answers_count[i,j] + 1

            
# This cell aims to save the auth_answers_count array in auth_answers_count.csv
np.savetxt("auth_answers_count.csv", auth_answers_count, fmt='%1u', delimiter=",")
#%%
# Read auth_answers_count from auth_answers_count.csv

# number_of_survey_taken[i] is the number of survey answered by all_auth_id_array[i]
number_of_survey_taken = np.sum(auth_answers_count, axis=1)
# number_of_participants_by_survey[i] is the number of participants to survey dfs[j]
number_of_participants_by_survey = np.sum(auth_answers_count, axis=0)
    