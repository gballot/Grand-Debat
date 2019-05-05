#%%
import numpy as np
import multiprocessing
from src.utils import read_data, get_open_reponses, get_ids_open_reponses
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
df_democratie = read_data('data/DEMOCRATIE_ET_CITOYENNETE.json')
df_resp_dem = get_open_reponses(df_democratie)
df_ids_dem = get_ids_open_reponses(df_democratie)
four_surveys_taken_auth_ids = np.loadtxt("four_surveys_taken_auth_ids.csv", delimiter=",", dtype=str)
ids_auth = np.sort(list(set(df_resp_dem['authorId'].values)))
np.savetxt("ids_auth_sorted.csv", ids_auth, delimiter=",", fmt="%s")
X = np.zeros((len(four_surveys_taken_auth_ids), n_compo))
# read features
features = np.loadtxt('responses democratie_all_questions.tsv', delimiter='\t')
# Fit GMM
gmm = GaussianMixture(n_components=n_compo)
gmm.fit(features)
# pool
local_pool = multiprocessing.Pool(10)
X = np.array(local_pool.map(fill_X, range(len(four_surveys_taken_auth_ids))))
local_pool.close()
local_pool.join()
np.savetxt("X_democratie.csv", X, delimiter=",")








