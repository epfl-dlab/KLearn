from functools import singledispatch
from scipy.stats import kendalltau
import numpy as np
import pickle
import io

from klearn import IT
# from klearn import Convertor

############################################################
# I/O utils
############################################################


class ConvertorUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "Convertor":
            renamed_module = "klearn.Convertor"

        return super(ConvertorUnpickler, self).find_class(renamed_module, name)


def convertor_load(file_obj):
    return ConvertorUnpickler(file_obj).load()


def convertor_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return convertor_load(file_obj)


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)


def load_data(filepath, use_lda=False):
    with open(filepath, 'rb') as f:
        # converted_dataset = pickle.loads(f.read())
        converted_dataset = convertor_loads(f.read())
    return converted_dataset


############################################################
# Evaluation utils
############################################################

def evaluate_topic(convertion, N=None, tgt='pyr_score'):
    annotations = convertion.annotations
    humans, freqs = [], []

    if N:
        annotations = sorted(annotations, key=lambda t: t[tgt], reverse=True)[:N]
    for summary in annotations:
        freqs.append(summary['freq'])
        humans.append(summary['pyr_score'])

    automatic = IT.theta_lst(convertion.doc_freq, freqs, convertion.K, convertion.alpha, convertion.beta)
    return kendalltau(humans, automatic)[0]


def evaluate_K(converted_dataset, K, alpha=1., beta=1., N=None, tgt='pyr_score'):
    correlations = []
    for topic_name, convertion in converted_dataset.items():
        annotations = convertion.annotations
        humans, freqs = [], []

        if N:
            annotations = sorted(annotations, key=lambda t: t[tgt], reverse=True)[:N]
        for summary in annotations:
            freqs.append(summary['freq'])
            humans.append(summary['pyr_score'])

        automatic = IT.theta_lst(convertion.doc_freq, freqs, K, alpha, beta)
        correlations.append(kendalltau(humans, automatic)[0])
    return correlations
    # return sum(correlations) / float(len(correlations))


def evaluate_MR(converted_dataset, K, alpha=1., beta=1., N=1, tgt='pyr_score'):
    MRs = []
    # observations = 0.
    for topic_name, convertion in converted_dataset.items():
        annotations = convertion.annotations
        humans, freqs = [], []

        sorted_annotations = sorted(annotations, key=lambda t: t[tgt], reverse=True)
        for summary in sorted_annotations:
            freqs.append(summary['freq'])
            humans.append(summary['pyr_score'])

        automatic = IT.theta_lst(convertion.doc_freq, freqs, K, alpha, beta)
        automatic_sorted = sorted(automatic, reverse=True)
        for i in range(N):
            # mr = automatic_sorted.index(automatic[i])
            MRs.append(automatic_sorted.index(automatic[i]))
            # observations += 1
    return MRs
    # return MRs / observations


def evaluate_JS_KL_baseline(converted_dataset, N=1, method='KL', tgt='pyr_score'):
    correlations, MRs = [], []

    for topic_name, convertion in converted_dataset.items():
        annotations = convertion.annotations
        humans, freqs = [], []

        sorted_annotations = sorted(annotations, key=lambda t: t[tgt], reverse=True)
        for summary in sorted_annotations:
            freqs.append(summary['freq'])
            humans.append(summary['pyr_score'])

        if method == 'KL':
            automatic = IT.kl_lst(convertion.doc_freq, freqs)
        else:
            automatic = IT.js_lst(convertion.doc_freq, freqs)
        correlations.append(kendalltau(humans, automatic)[0])

        automatic_sorted = sorted(automatic, reverse=True)
        for i in range(N):
            MRs.append(automatic_sorted.index(automatic[i]))

    return MRs, correlations

# def load_data(filepath):
#     with open(filepath, 'rb') as f:
#         return pickle.loads(f.read())
