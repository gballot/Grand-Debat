# -*- coding: utf-8 -*-

import fastText
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import stop_words
import pathlib


def tokenize(text):
    return word_tokenize(text, language='french')


class FeaturesExtractor:
    """ Handle features extractions based on word embeddings (fasttext) """
    def __init__(self,
                 model_path: str = 'data/cc.fr.300.bin'):
        assert model_path.endswith('.bin'), 'model_path should be a .bin file'
        assert pathlib.Path(model_path).exists(), 'model_path does not exists'

        self.stop_words = set(stopwords.words('french') +
                              list(string.punctuation) +
                              stop_words.get_stop_words('fr'))

        print(('loading model could take a while...'
               ' and takes up to 7GO of RAM'))
        self.model = fastText.load_model(model_path)

    def get_features(self, response: str):
        """
        """
        assert type(response) == str, 'response must be a string'
        words = tokenize(response)

        words = [x for x in words if x not in self.stop_words]

        return self.model.get_sentence_vector(' '.join(words))


# if __name__ == "__main__":
#     s = FeaturesExtractor()
#     features = [s.get_features(x) for x in responses]
#     from sklearn.cluster import KMeans

#     k = KMeans(n_clusters=15)
#     k.fit(np.array(features))

#     df = pd.DataFrame({'label': k.labels_, 'response': responses})

#     for label in df.label.unique():
#         print('label {}'.format(label))
#         samples = [x for x in df[df.label==label].sample(10).response.tolist()]
#         for sample in samples:
#             print(sample)
#         print('#'*20)
