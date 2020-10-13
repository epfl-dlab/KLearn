import klearn
import json
import os
import numpy as np
import random

random.seed(123)

OUTPUT_FOLDER = 'outputs/'

PREPROCESSING = ['1']
LEARNERS = [
    (klearn.algorithms.baseline, []),
    (klearn.algorithms.MS_U, []),
    (klearn.algorithms.IDF_baseline, []),
    (klearn.algorithms.MS_D, []),
    (klearn.algorithms.hPL, [10]),
]

DATASETS = {
    'TAC2008': klearn.utils.load_data('/Users/peyrardm/Documents/NLP/KLearn-preprocess/data_converted/TAC2008_1.pkl'),
    'TAC2009': klearn.utils.load_data('/Users/peyrardm/Documents/NLP/KLearn-preprocess/data_converted/TAC2009_1.pkl')
}
TGT = 'pyr_score'


def run_CV_experiment(dataset_folds, method, results, dataset_filename, nb_epochs=None):
    tau_eval, MR_eval = [], []
    for i, fold in enumerate(dataset_folds):
        test_fold = fold
        train_folds = [fold for j, fold in enumerate(dataset_folds) if j != i]
        # train_fold = list(itertools.chain.from_iterable(train_fold))
        train_fold = {k: v for d in train_folds for k, v in d.items()}

        if nb_epochs:
            parameters = method(train_fold, nb_epochs, tgt=TGT)
        else:
            parameters = method(train_fold, tgt=TGT)

        K_generic, alpha, beta = parameters
        tau_eval.extend(klearn.utils.evaluate_K(test_fold, K_generic, alpha, beta))
        MR_eval.extend(klearn.utils.evaluate_MR(test_fold, K_generic, alpha, beta, N=4))

    results.append({'learner': method.__name__,
                    'nb_epochs': nb_epochs,
                    'performance-tau': np.mean(tau_eval),
                    'MR-ref': np.mean(MR_eval),
                    'dataset': dataset_filename
                    })
    return results


def chunk_lst(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def cross_validation_split(converted_data, n_folds=4):
    idx_folds = chunk_lst(list(converted_data.keys()), n_folds)
    folds = []
    for idx_f in idx_folds:
        folds.append({k: v for k, v in converted_data.items() if k in idx_f})
    return folds


def add_JS_KL_baselines(converted_data, dataset_filename, results, N=4):
    MR_eval, tau_eval = klearn.utils.evaluate_JS_KL_baseline(converted_data, N=4, method='JS')

    results.append({'learner': 'JS(S||D)',
                    'performance-tau': np.mean(tau_eval),
                    'MR-ref': np.mean(MR_eval),
                    'dataset': dataset_filename
                    })

    MR_eval, tau_eval = klearn.utils.evaluate_JS_KL_baseline(converted_data, N=4, method='KL')

    results.append({'learner': 'KL(S||D)',
                    'performance-tau': np.mean(tau_eval),
                    'MR-ref': np.mean(MR_eval),
                    'dataset': dataset_filename
                    })

    return results


if __name__ == '__main__':
    results = []
    for dataset_name, converted_data in DATASETS.items():
        dataset_filename = dataset_name + '_1.pkl'

        print("DATASET: {}".format(dataset_name))

        converted_data_folds = cross_validation_split(converted_data)

        # for a in converted_data_folds:
        #     print(a.keys())

        for method, config in LEARNERS:
            if len(config) > 0:
                for nb_epochs in config:
                    print('LEARNER: {} - {}'.format(method.__name__, nb_epochs))
                    results = run_CV_experiment(converted_data_folds, method, results, dataset_filename, nb_epochs)
            else:
                print('LEARNER: {}'.format(method.__name__))
                results = run_CV_experiment(converted_data_folds, method, results, dataset_filename, nb_epochs=None)

        results = add_JS_KL_baselines(converted_data, dataset_filename, results)

    # with open(os.path.join(OUTPUT_FOLDER, 'model_comparison_cv.json'), 'w') as f:
    #     f.write(json.dumps(results))
