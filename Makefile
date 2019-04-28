default:

all: update_data import_tsv_from_omv get_fastText nltk

update_data:
	$(RM) data/LA_TRANSITION_ECOLOGIQUE.json data/LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES.json data/DEMOCRATIE_ET_CITOYENNETE.json data/ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS.json
	wget http://opendata.auth-6f31f706db6f4a24b55f42a6a79c5086.storage.sbg5.cloud.ovh.net/2019-02-06/LA_TRANSITION_ECOLOGIQUE.json -P data/
	wget http://opendata.auth-6f31f706db6f4a24b55f42a6a79c5086.storage.sbg5.cloud.ovh.net/2019-02-06/LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES.json -P data/
	wget http://opendata.auth-6f31f706db6f4a24b55f42a6a79c5086.storage.sbg5.cloud.ovh.net/2019-02-06/DEMOCRATIE_ET_CITOYENNETE.json -P data/
	wget http://opendata.auth-6f31f706db6f4a24b55f42a6a79c5086.storage.sbg5.cloud.ovh.net/2019-02-06/ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS.json -P data/
	wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz -P data/
	cd data; gzip -d cc.fr.300.bin.gz

import_tsv_from_omv:
	scp root@90.92.106.62:Grand-Debat/*.tsv .

get_fastText:
	cd; git clone https://github.com/facebookresearch/fastText.git; cd fastText; make; pip3 install .

nltk:
	pip3 install nltk
	python3 -c "import nltk;nltk.download('stopwords');nltk.download('punkt')"

