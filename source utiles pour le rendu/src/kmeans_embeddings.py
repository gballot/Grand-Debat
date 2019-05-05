# -*- coding: utf-8 -*-

import fastText
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
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
        self.porter = PorterStemmer()

    def get_features(self, response: str):
        """
        """
        assert type(response) == str, 'response must be a string'
        words = tokenize(response)
        # this line deletes the stopwords, keeps alphanumeric only, put the case to lower
        # and stems the words
        words = [self.porter.stem(x.lower()) for x in words if x not in self.stop_words and x.isalpha()]

        return self.model.get_sentence_vector(' '.join(words))