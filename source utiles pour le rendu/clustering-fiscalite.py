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
df_fiscalite = read_data('data/LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES.json')
df_resp_fis = get_open_reponses(df_fiscalite)
df_ids_fis = get_ids_open_reponses(df_fiscalite)
four_surveys_taken_auth_ids = np.loadtxt("four_surveys_taken_auth_ids.csv", delimiter=",", dtype=str)
ids_auth = np.sort(list(set(df_resp_fis['authorId'].values)))
np.savetxt("ids_auth_sorted.csv", ids_auth, delimiter=",", fmt="%s")
X = np.zeros((len(four_surveys_taken_auth_ids), n_compo))
# read features
features = np.loadtxt('responses fiscalite_all_questions.tsv', delimiter='\t')
# Fit GMM
gmm = GaussianMixture(n_components=n_compo)
gmm.fit(features)
# pool
local_pool = multiprocessing.Pool(10)
X = np.array(local_pool.map(fill_X, range(len(four_surveys_taken_auth_ids))))
local_pool.close()
local_pool.join()
np.savetxt("X_fiscalite.csv", X, delimiter=",")








