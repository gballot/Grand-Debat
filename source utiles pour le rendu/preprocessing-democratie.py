#%%
import numpy as np
from src.kmeans_embeddings import FeaturesExtractor
from src.utils import read_data, get_open_reponses, get_ids_open_reponses

#%% extract data from json
df_fiscalite = 0
df_democratie = read_data('data/DEMOCRATIE_ET_CITOYENNETE.json')
df_ecologie = 0
df_organisation = 0

dfs = np.array([["fiscalite", df_fiscalite], ["democratie", df_democratie], ["ecologie", df_ecologie], ["organisation", df_organisation]])
#%%

#%% responses of each themes
df_resp_fis = 0
df_resp_dem = get_open_reponses(df_democratie)
df_resp_eco = 0
df_resp_org = 0

dfs_responses = np.array([["responses fiscalite", df_resp_fis], ["responses democratie", df_resp_dem], ["responses ecologie", df_resp_eco], ["responses organisation", df_resp_org]])
#%%

#%% extract features
s = FeaturesExtractor()
def extract_features():
    for k in [1]:
        ids_questions = get_ids_open_reponses(dfs[k,1])
        ids_auth = np.sort(list(set(dfs_responses[k,1]['authorId'].values)))
        # Extract embeddings for sentences
        features = np.zeros((len(ids_auth), 300*len(ids_questions)))
        for i in range(len(ids_auth)) :
            for j in range(len(ids_questions)) :
                response_unique = dfs_responses[k,1][dfs_responses[k,1]['authorId'] == ids_auth[i]][dfs_responses[k,1][dfs_responses[k,1]['authorId'] == ids_auth[i]]['questionId'] == ids_questions[j]].formattedValue.values.tolist()
                if (len(response_unique) > 0) :
                    features[i][300*j:300*(j+1)] = s.get_features(response_unique[0])
                else:
                    features[i][300*j:300*(j+1)] = [0.]*300
        np.savetxt(dfs_responses[k,0,]+'_all_questions.tsv', features, delimiter='\t')
        
#%%

extract_features()

