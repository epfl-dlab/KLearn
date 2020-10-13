import numpy as np


def normalize(distribution):
    min_val = min(distribution.values())
    if min_val < 0:
        distribution = dict([(k, v - min_val) for k, v in distribution.items()])

    sum_val = sum(distribution.values())
    return dict([(k, v / sum_val) for k, v in distribution.items()])


def avg(distributions):
    all_keys = [d.keys() for d in distributions]
    keys = list(set().union(*all_keys))
    M = {}
    for key in keys:
        M[key] = np.mean([d.get(key, 0) for d in distributions])
    return M


def Entropy(distribution):
    H = 0
    for v in distribution.values():
        if v > 0:
            H += v * np.log(1. / v)
    return H


def Cross_Entropy(P, Q):
    CE = 0
    for key, v in Q.items():
        if v > 0:
            CE += P.get(key, 0) * np.log(1. / v)
    return CE


def KL(P, Q):
    kl = 0
    for key, v in Q.items():
        if v > 0:
            other_v = P.get(key, 0)
            if other_v > 0:
                kl += other_v * np.log(other_v / v)
    return kl


def JS(P, Q):
    M = {}
    all_keys = set(P.keys()).union(set(Q.keys()))
    for key in all_keys:
        M[key] = (P.get(key, 0) + Q.get(key, 0)) / 2.
    return (KL(P, M) + KL(Q, M)) / 2.


def theta_lst(doc_freq, lst_freqs, K, alpha=-1, beta=1):
    results = []
    for f in lst_freqs:
        results.append(Entropy(f) - alpha * KL(f, doc_freq) + beta * KL(f, K))
    return results


def theta(doc_freq, freqs, K, alpha, beta):
    return Entropy(freqs) - alpha * KL(freqs, doc_freq) + beta * KL(freqs, K)


def js_lst(doc_freq, lst_freqs):
    results = []
    for f in lst_freqs:
        results.append(-JS(doc_freq, f))
    return results


def kl_lst(doc_freq, lst_freqs):
    results = []
    for f in lst_freqs:
        results.append(-KL(doc_freq, f))
    return results
