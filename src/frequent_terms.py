# -*- coding: utf-8 -*-

import itertools
import string
from collections import Counter
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from stop_words import get_stop_words


x = df_open[df_open.questionId == '107']

answers = x.formattedValue.values.tolist()
answers = ' '.join(answers)
answers = answers.lower()

stop_words = set(stopwords.words('french') +
                 list(string.punctuation) +
                 get_stop_words('fr'))
word_tokens = word_tokenize(answers, language='french')

words = [x for x in word_tokens if x not in stop_words]

cnt = Counter(words)
cnt.most_common(20)
