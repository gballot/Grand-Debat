all:


update_data:
	$(RM) data/LA_TRANSITION_ECOLOGIQUE.json data/LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES.json data/DEMOCRATIE_ET_CITOYENNETE.json data/ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS.json
	wget http://opendata.auth-6f31f706db6f4a24b55f42a6a79c5086.storage.sbg5.cloud.ovh.net/2019-02-06/LA_TRANSITION_ECOLOGIQUE.json -P data/
	wget http://opendata.auth-6f31f706db6f4a24b55f42a6a79c5086.storage.sbg5.cloud.ovh.net/2019-02-06/LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES.json -P data/
	wget http://opendata.auth-6f31f706db6f4a24b55f42a6a79c5086.storage.sbg5.cloud.ovh.net/2019-02-06/DEMOCRATIE_ET_CITOYENNETE.json -P data/
	wget http://opendata.auth-6f31f706db6f4a24b55f42a6a79c5086.storage.sbg5.cloud.ovh.net/2019-02-06/ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS.json -P data/
