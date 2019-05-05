import src.utils as ut
import numpy as np
import pandas as pd
import string
import multiprocessing
from src.utils import (read_data, get_open_reponses, get_ids_open_reponses)
from som_batched_learning import (open_model, get_clusters)
from X import(get_X, get_auth_id)

#Output of the first GMM learning stage   
X = get_X()
#Output of the 2nd stage: best SOM model found after 500 models trained
best_som_model = open_model(60)

clusters = get_clusters(nb_clusters=10, X_projected=X, sm=best_som_model)

df_fiscalite = ut.read_data('data/LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES.json')
df_democratie = ut.read_data('data/DEMOCRATIE_ET_CITOYENNETE.json')
df_ecologie = ut.read_data('data/LA_TRANSITION_ECOLOGIQUE.json')
df_organisation = ut.read_data('data/ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS.json')
dfs = np.array([["fiscalite", df_fiscalite], ["democratie", df_democratie], ["ecologie", df_ecologie], ["organisation", df_organisation]])

df_reponses = get_open_reponses(dfs)

print(clusters)

