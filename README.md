# Installation

```bash
git clone
cd 
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
```

Installation fastText:

```
git clone https://github.com/facebookresearch/fastText
cd fastText
python setup.py install
```

Téléchargement des données du Grand Debat:


```bash
wget
http://opendata.auth-6f31f706db6f4a24b55f42a6a79c5086.storage.sbg5.cloud.ovh.net/2019-02-06/LA_TRANSITION_ECOLOGIQUE.json
-P data/
wget
http://opendata.auth-6f31f706db6f4a24b55f42a6a79c5086.storage.sbg5.cloud.ovh.net/2019-02-06/LA_FISCALITE_ET_LES_DEPENSES_PUBLIQUES.json
-P data/
wget
http://opendata.auth-6f31f706db6f4a24b55f42a6a79c5086.storage.sbg5.cloud.ovh.net/2019-02-06/DEMOCRATIE_ET_CITOYENNETE.json
-P data/
wget
http://opendata.auth-6f31f706db6f4a24b55f42a6a79c5086.storage.sbg5.cloud.ovh.net/2019-02-06/ORGANISATION_DE_LETAT_ET_DES_SERVICES_PUBLICS.json
-P data/
```

Téléchargement données pour FastText (word embeddings)

Download French pre-trained model at: https://fasttext.cc/docs/en/crawl-vectors.html

# docs
https://nbviewer.jupyter.org/github/rare-technologies/gensim/blob/develop/docs/notebooks/atmodel_tutorial.ipynb
