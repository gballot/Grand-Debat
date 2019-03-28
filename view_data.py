import src.utils as ut
import pandas as pd

#%% extract data from json

df_LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES = ut.read_data('data/LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES.json')
df_DEMOCRATIE_ET_CITOYENNETE = ut.read_data('data/DEMOCRATIE_ET_CITOYENNETE.json')
df_LA_TRANSITION_ECOLOGIQUE = ut.read_data('data/LA_TRANSITION_ECOLOGIQUE.json')
df_ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS = ut.read_data('data/ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS.json')

#%%
df_LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES['authorId'][0]
auth='VXNlcjo3ZTVjYTUwMi0xZDZlLTExZTktOTRkMi1mYTE2M2VlYjExZTE='
df_LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES.columns

#%%
def extract_responses_by_authorId(df: pd.DataFrame, key: str='VXNlcjo3ZTVjYTUwMi0xZDZlLTExZTktOTRkMi1mYTE2M2VlYjExZTE='):
    """ Extract a specific question
 
    Args:
        df: dataframe
        key: authorId (example 'VXNlcjo3ZTVjYTUwMi0xZDZlLTExZTktOTRkMi1mYTE2M2VlYjExZTE=')
 
    Returns:
        responses as a string
    """

    df = [df.loc[i,'formattedValue'] for i in range(df.shape[0])
                if df.loc[i,'authorId'] == key]
    if len(fd):
        return fd[0]
    else:
        return None
    
#%%
extract_responses_by_authorId(df_DEMOCRATIE_ET_CITOYENNETE)

#%%

df = df_LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES
#print(df['authorID'])
df.loc[df['authorId'] == auth]

#%%
dfs = [df_LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES, df_DEMOCRATIE_ET_CITOYENNETE, df_LA_TRANSITION_ECOLOGIQUE, df_ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS]

for df in dfs:
    print(df.loc[df['authorId'] == auth])