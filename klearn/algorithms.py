import numpy as np
import random
import pickle

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from klearn import IT
from klearn import KModels


def unique_key_set(dataset):
    dataset_keys = []
    for k, v in dataset.items():
        dataset_keys.extend(v.all_keys())

    return list(set(dataset_keys))


def restrict_key_set(dataset, unique_dataset_keys, top_k):
    count_keys = {}
    for k in unique_dataset_keys:
        count_keys[k] = sum(doc.doc_freq.get(k, 0) for _, doc in dataset.items())

    unique_dataset_keys = sorted(count_keys.items(), key=lambda t: t[1], reverse=True)[:top_k]
    return sorted([x for x, y in unique_dataset_keys])


def cut_off(K, nb_cut):
    if nb_cut >= len(K):
        return K
    nb_keep = len(K) - nb_cut
    new_K = dict(sorted(K.items(), key=lambda t: t[1], reverse=True)[:nb_keep])
    return IT.normalize(new_K)


def baseline(dataset, tgt=None, restriction=None):
    key_set = unique_key_set(dataset)
    if restriction:
        key_set = restrict_key_set(dataset, key_set, top_k=restriction)
    return dict((x, 1. / len(key_set)) for x in key_set), 1., 1.


def load_idfs():
    with open('data/idfs', 'rb') as f:
        idfs = pickle.loads(f.read())

    # idfs = dict([(k,v) for k,v in idfs.items() if k not in stopset])
    K = np.array(list(idfs.values()))
    K = - K
    K -= np.min(K)
    K = K / np.sum(K)
    return dict(zip(idfs.keys(), K))


def IDF_baseline(dataset, tgt=None, restriction=None):
    key_set = unique_key_set(dataset)
    if restriction:
        key_set = restrict_key_set(dataset, key_set, top_k=restriction)

    return load_idfs(), 1., 1.


def MS_U(dataset, tgt='pyr_score', restriction=None):
    unique_dataset_keys = sorted(unique_key_set(dataset))

    if restriction:
        unique_dataset_keys = restrict_key_set(dataset, unique_dataset_keys, top_k=restriction)

    S_all = np.zeros(len(unique_dataset_keys))
    for k, v in dataset.items():
        v.normalize_keysets(unique_dataset_keys)

        try:
            kept = sorted(v.annotations, key=lambda t: t[tgt], reverse=True)[:4]
        except Exception:
            kept = v.annotations

        for annot in kept:
            vec = np.array([annot['freq'][j] for j in unique_dataset_keys])
            vec_dis = [(1 - v) for v in vec]
            S_all += vec_dis

    # K_all = 1 - S_all
    # K_all -= np.min(K_all)
    K_all = S_all / np.sum(S_all)

    return dict(zip(unique_dataset_keys, K_all.tolist())), 1., 1.


def MS_D(dataset, tgt='pyr_score', restriction=None):
    unique_dataset_keys = sorted(unique_key_set(dataset))

    if restriction:
        unique_dataset_keys = restrict_key_set(dataset, unique_dataset_keys, top_k=restriction)

    S_all = np.zeros(len(unique_dataset_keys))
    D_all = np.zeros(len(unique_dataset_keys))
    for k, v in dataset.items():
        v.normalize_keysets(unique_dataset_keys)
        D_all += np.array([v.doc_freq[j] for j in unique_dataset_keys])

        # Code to accomodate different datasets
        nb_keep = min(4, len(v.annotations))
        if tgt in v.annotations[0]:
            kept = sorted(v.annotations, key=lambda t: t[tgt], reverse=True)[:nb_keep]
        else:
            kept = v.annotations

        for annot in kept:
            vec = np.array([annot['freq'][j] for j in unique_dataset_keys])
            S_all += vec

    gamma = np.min(np.divide(S_all, D_all))

    K_all = gamma * D_all - S_all  # - 0.0001
    K_all -= np.min(K_all)
    K_all = K_all / np.sum(K_all)

    return dict(zip(unique_dataset_keys, K_all)), 1., 1.


def extract_tensors(dataset, unique_dataset_keys, tgt):
    T_all, S_all, Y_all = {}, {}, {}
    for k, v in dataset.items():
        v.normalize_keysets(unique_dataset_keys)
        v.smooth(v.min_freq() / 10.)
        t, S, Y = v.get_vectors(tgt, unique_dataset_keys)
        T = np.repeat(t.reshape(1, len(t)), repeats=S.shape[0], axis=0)
        Y = Y / np.max(Y)

        T_all[k] = T
        S_all[k] = S
        Y_all[k] = Y

    return T_all, S_all, Y_all


def get_batch_PL(T_all, S_all, Y_all, batch_size):
    T, S_1, S_2, Y_class = [], [], [], []
    keys = list(T_all.keys())
    for _ in range(batch_size):
        key = random.choice(keys)

        idx = random.sample(range(S_all[key].shape[0]), 2)
        idx_a = idx[0]
        idx_b = idx[1]
        S_1.append(S_all[key][idx_a, :])
        S_2.append(S_all[key][idx_b, :])
        Y_class.append(float(Y_all[key][idx_a] > Y_all[key][idx_b]))

        T.append(T_all[key][0])

    S_1 = np.vstack(S_1)
    S_2 = np.vstack(S_2)
    T = np.vstack(T)
    Y_class = np.vstack(Y_class)

    return torch.from_numpy(T).float(), torch.from_numpy(S_1).float(), torch.from_numpy(S_2).float(), torch.from_numpy(Y_class).float()


def hPL(dataset, nb_epochs, tgt='pyr_score', lr=100, batch_size=2048, restriction=None):
    unique_dataset_keys = sorted(unique_key_set(dataset))

    if restriction:
        unique_dataset_keys = restrict_key_set(dataset, unique_dataset_keys, top_k=restriction)

    T_all, S_all, Y_all = extract_tensors(dataset, unique_dataset_keys, tgt)

    torch.autograd.set_detect_anomaly(True)

    # initial_K, _, _ = MS_D(dataset)

    theta = KModels.ThetaPL(len(unique_dataset_keys))  # , list(initial_K.values()))
    criterion = nn.BCELoss()
    optimizer = optim.SGD([theta.k], lr=lr)

    # nb_epochs = 3000
    for _ in range(nb_epochs):
        T_training, S1, S2, Y_training = get_batch_PL(T_all, S_all, Y_all, batch_size)

        optimizer.zero_grad()   # zero the gradient buffers
        output = theta(T_training, S1, S2)
        loss = criterion(output, Y_training)

        print("loss: {}".format(loss))
        loss.backward()
        optimizer.step()

    K = dict(zip(unique_dataset_keys, F.softmax(torch.from_numpy(theta.k.data.numpy()), dim=0).numpy()))

    return K, 1., 1.
