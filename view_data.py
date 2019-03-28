import src.utils as ut

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
def extract_responses_by_authorId(responses: list, key: str='VXNlcjo3ZTVjYTUwMi0xZDZlLTExZTktOTRkMi1mYTE2M2VlYjExZTE='):
     """ Extract a specific question
 
     Args:
         responses: list (example df.iloc[0].responses)
         key: authorId (example 'VXNlcjo3ZTVjYTUwMi0xZDZlLTExZTktOTRkMi1mYTE2M2VlYjExZTE=')
 
     Returns:
         responses as a string
     """
 
     response = [x['formattedValue'] for x in responses
                 if x['authorId'] == key]
     if len(response):
         return response[0]
     else:
         return None
    
#%%
extract_responses_by_authorId(df_DEMOCRATIE_ET_CITOYENNETE.iloc[0].responses)