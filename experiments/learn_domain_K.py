import json
import sys
import os

sys.path.insert(0, os.path.abspath('..'))

from klearn import utils
from klearn import algorithms


def write_K(K, filepath):
    with open(filepath, 'w') as f:
        f.write(json.dumps(K))


DOMAINS_K_FOLDER = '../outputs/'
DATA_PATH = '../data_converted/'
DATASETS = [
    ('PubMed', os.path.join(DATA_PATH, 'PubMed_1.pkl')),
    ('Opinosis', os.path.join(DATA_PATH, 'Opinosis_1.pkl')),
    ('hMDS', os.path.join(DATA_PATH, 'hMDS_1.pkl')),
    ('SciSumm', os.path.join(DATA_PATH, 'SciSumm_1.pkl')),
    ('AMI', os.path.join(DATA_PATH, 'AMI_1.pkl')),
    ('Reddit', os.path.join(DATA_PATH, 'Reddit_1.pkl')),
    ('CNN-DM', os.path.join(DATA_PATH, 'CNN-DM_1.pkl')),
    ('X-Sum', os.path.join(DATA_PATH, 'X-Sum_1.pkl')),
    ('MDIC', os.path.join(DATA_PATH, 'MDIC_1.pkl')),
    ('NYT', os.path.join(DATA_PATH, 'NYT_1.pkl')),
    ('WikiHow', os.path.join(DATA_PATH, 'WikiHow_1.pkl')),
    ('LegalReports', os.path.join(DATA_PATH, 'LegalReports_1.pkl')),
    ('LiveBlogs', os.path.join(DATA_PATH, 'LiveBlogs_1.pkl')),
    ('TAC2009', os.path.join(DATA_PATH, 'TAC2009_1.pkl')),
    ('TAC2008', os.path.join(DATA_PATH, 'TAC2008_1.pkl'))
]

results = {}
for dataset, filepath in DATASETS:
    print(dataset)

    print('Loading: ', end='')
    sys.stdout.flush()
    converted_dataset = utils.load_data(filepath)
    if len(converted_dataset) > 5000:
        converted_dataset = dict((k, v) for k, v in list(converted_dataset.items())[:5000])
    print('[ok]')

    print('Finding K: ', end='')
    sys.stdout.flush()
    parameters = algorithms.MS_U(converted_dataset, restriction=8000)  # THINK ABOUT APPLYING NORMALIZATION
    print('[ok]')

    print('Writing K: ', end='')
    sys.stdout.flush()
    K, alpha, beta = parameters
    results[dataset] = {'K': K}
    print('[ok]')

with open(os.path.join(DOMAINS_K_FOLDER, 'domain_Ks.json'), 'w') as f:
    f.write(json.dumps(results, default=utils.to_serializable))
