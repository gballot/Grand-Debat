# -*- coding: utf-8 -*-


from src.kmeans_embeddings import FeaturesExtractor
from src.utils import (read_data, get_open_reponses)
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


if __name__ == '__main__':
    df = read_data('data/DEMOCRATIE_ET_CITOYENNETE.json')
    df_responses = get_open_reponses(df)

    responses = (df_responses[df_responses.questionId == '107'].
                 formattedValue.values.tolist())

    # Extract embeddings for sentences
    s = FeaturesExtractor()
    features = [s.get_features(x) for x in responses]

    features_np = np.array(features)

    samples_id = np.random.choice(range(len(features)), 5000)

    features_np_samples = features_np[samples_id, :]
    np.savetxt('features_s.tsv', features_np_samples, delimiter='\t')
    responses_samples = [responses[i] for i in samples_id]
    with open('labels_s.tsv', 'w') as f:
        for resp in responses_samples:
            v = resp.replace('\n', '. ')
            v = v.replace('\t', '. ')
            f.write('{}\n'.format(v))
    # Fit Kmeans
    k = KMeans(n_clusters=15)
    k.fit(np.array(features))

    # print samples from each clusters
    df = pd.DataFrame({'label': k.labels_, 'response': responses})

    for label in df.label.unique():
        print('label {}'.format(label))
        samples = [x for x in df[df.label == label].sample(10).response.tolist()]
        for sample in samples:
            print(sample)
        print('#'*20)
