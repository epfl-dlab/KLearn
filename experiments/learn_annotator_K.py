import json
import os

import sys
sys.path.insert(0, os.path.abspath('..'))

from klearn import utils
from klearn import algorithms

ANNOTATORS_K_FOLDER = '../outputs/'
ANNOTATORS_TOPICS = 'assessor_topics.json'
INPUT_FOLDER = '../data'


def write(K, filepath):
    K_serializable = K
    if type(list(K.keys())[0]) != str:
        K_serializable = dict((k, v) for k, v in
                              zip([str(tup) for tup in K.keys()], K.values()))
    with open(filepath, 'w') as f:
        f.write(json.dumps(K_serializable, default=utils.to_serializable))


with open(ANNOTATORS_TOPICS, 'r') as f:
    assessors = json.loads(f.read())

list_assessors = list(set(assessors.values()))

dataset = utils.load_data(os.path.join(INPUT_FOLDER, 'TAC2008_1.pkl'))
dataset_09 = utils.load_data(os.path.join(INPUT_FOLDER, 'TAC2009_1.pkl'))
dataset.update(dataset_09)

results = {}
for assessor in list_assessors:
    restricted_dataset = dict([(k, v) for k, v in dataset.items() if assessors[k] == assessor])

    parameters = algorithms.hPL(restricted_dataset, nb_epochs=200)
    K_assessor, alpha, beta = parameters
    results[assessor] = {'K': K_assessor, 'fit': utils.evaluate_K(restricted_dataset, K_assessor, alpha, beta)}

with open(os.path.join(ANNOTATORS_K_FOLDER, 'assessors_Ks.json'), 'w') as f:
    f.write(json.dumps(results, default=utils.to_serializable))
