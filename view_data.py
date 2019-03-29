# librairies imported
import src.utils as ut
import numpy as np
import pandas as pd

#%% extract data from json
df_fiscalite = ut.read_data('data/LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES.json')
df_democratie = ut.read_data('data/DEMOCRATIE_ET_CITOYENNETE.json')
df_ecologie = ut.read_data('data/LA_TRANSITION_ECOLOGIQUE.json')
df_organisation = ut.read_data('data/ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS.json')
#%%
np.random.seed(1)
for i in np.random.randint(len(df_fiscalite), size=5):
    auth = df_fiscalite.loc[i, 'authorId']
    print("Author ID : " + auth)

    dfs = np.array([["fiscalite", df_fiscalite], ["democratie", df_democratie], ["ecologie", df_ecologie], ["organisation", df_organisation]])
    for df in dfs:
        code = df[1].loc[df[1]['authorId'] == auth, 'authorZipCode']
        if(len(code) > 0):
            code = code.values[0]
            print("* In " + df[0] + " survey, author has zip code : " + str(code))
        else:
            print("* In " + df[0] + " survey, author has not answered...")
    print("\n############################\n")

#%% Be careful, this cell takes a lot of time to run !!!!!
    
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
#%%
# Read auth_answers_count from auth_answers_count.csv
auth_answers_count = np.loadtxt('auth_answers_count.csv', dtype=int ,delimiter=",")

print("auth_answers_count :")
print(auth_answers_count)

# number_of_survey_taken[i] is the number of survey answered by all_auth_id_array[i]
number_of_survey_taken = np.sum(auth_answers_count, axis=1)
# number_of_participants_by_survey[i] is the number of participants to survey dfs[j]
number_of_participants_by_survey = np.sum(auth_answers_count, axis=0)

print("#######################")
print("number of participant by survey :")
for i in range(4):
    print(dfs[i,0] + " : " + str(number_of_participants_by_survey[i]))

# number_of_participant_to_several_surveys[i] is the number of participants that have
# answerd to i surveys out of the 4 (0<i<5)
number_of_participant_to_several_surveys = np.bincount(number_of_survey_taken)

print("#######################")
print("number of participant to x surveys :")
for i in range(5):
    print(str(number_of_participant_to_several_surveys[i]) + " people have participed to "
          + str(i) + " different surveys.")
    
#%%
