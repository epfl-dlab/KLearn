import json
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

from klearn import utils
from klearn import algorithms


def write_K(K, filepath):
    with open(filepath, 'w') as f:
        f.write(json.dumps(K, default=utils.to_serializable))


DOMAINS_K_FOLDER = '../outputs/'
DATA_PATH = '../data_converted/'
DATASETS = {
    'TAC2009-1': os.path.join(DATA_PATH, 'TAC2009_1.pkl'),
    'TAC2008-1': os.path.join(DATA_PATH, 'TAC2008_1.pkl'),
    'TAC2009-40': os.path.join(DATA_PATH, 'TAC2009_LDA_model_40.pkl'),
    'TAC2008-40': os.path.join(DATA_PATH, 'TAC2008_LDA_model_40.pkl'),
    # 'TAC2008-200': os.path.join(DATA_PATH, 'TAC2008_LDA_model_200.pkl')
}

SUPPORTS = ['1', '40']


def optimal_K(dataset, support):
    results = []
    for i in range(10):
        print('Finding K: ', end='')
        sys.stdout.flush()
        if support != '1':
            parameters = algorithms.hPL(dataset, nb_epochs=500, lr=1, restriction=2000)
        else:
            parameters = algorithms.hPL(dataset, nb_epochs=2000, restriction=2000)
        print('[ok]')

        print('Writing K: ', end='')
        sys.stdout.flush()
        K, alpha, beta = parameters
        results.append(K)
        print('[ok]')

    write_K(results, os.path.join('../outputs', 'optimal_K_' + support + '_.json'))


if __name__ == '__main__':
    support = sys.argv[1]

    assert support in SUPPORTS, \
        "support not in {}".format(SUPPORTS)

    print('Loading: ', end='')
    sys.stdout.flush()
    if support == '1':
        dataset = utils.load_data(DATASETS['TAC2008-1'])
        dataset.update(utils.load_data(DATASETS['TAC2009-1']))
    elif support == '40':
        dataset = utils.load_data(DATASETS['TAC2008-40'])
        dataset.update(utils.load_data(DATASETS['TAC2009-40']))
    else:
        dataset = utils.load_data(DATASETS['TAC2008-200'])
    print('[ok]')

    optimal_K(dataset, support)
