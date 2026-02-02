import numpy as np
from scipy.ndimage import gaussian_filter


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

def kl(p, q):
    return np.nansum(p * (np.log2(p + 1e-10) - np.log2(q + 1e-10)))

def estimate_maximum_likelihood_transitions(sequence, unique_states=None):
    import itertools
    unique_states = np.unique(sequence) if unique_states is None else unique_states
    dim = len(unique_states)
    transitions = {token: 0 for token in itertools.product(unique_states, unique_states)}
    for x0, x1 in zip(sequence[:-1], sequence[1:]):
        transitions[(x0, x1)] += 1

    matrix = np.zeros((dim, dim))

    for i, s_from in enumerate(unique_states):
        for j, s_to in enumerate(unique_states):
            matrix[i, j] = transitions[(s_from, s_to)]

    # Normalize rows
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        est = np.divide(matrix, row_sums, where=row_sums > 0)
    return est

def sensor_eye(dim, gaussian=None):
    if gaussian is None:
        return np.eye(dim)

    if gaussian.get('mode') is not None:
        gaussian['mode'] = 'reflect'
    return gaussian_filter(np.eye(dim), sigma=gaussian.get('sigma'), mode=gaussian.get('mode'))

def entropy(P):
    P = P[P>0]  # prevent zero log
    return - np.sum(P*np.log2(P))

def jsd(P, Q):
    H_pq = entropy(0.5*(P+Q))
    H_p = entropy(P)
    H_q = entropy(Q)
    return H_pq - 0.5*H_p - 0.5*H_q

def jsd_avg(P, Q):
    jsds = []
    for p, q in zip(P, Q):
        H_pq = entropy(0.5*(p+q))
        H_p = entropy(p)
        H_q = entropy(q)
        jsds.append(H_pq - 0.5*H_p - 0.5*H_q)
    return np.mean(jsds)

def jsd_sum(P, Q):
    jsds = []
    for p, q in zip(P, Q):
        H_pq = entropy(0.5*(p+q))
        H_p = entropy(p)
        H_q = entropy(q)
        jsds.append(H_pq - 0.5*H_p - 0.5*H_q)
    return np.sum(jsds)

def get_pi(W):
    w, v = np.linalg.eig(W)
    idx = np.argmin(np.abs(w - 1.0))
    v1 = np.real(v[:, idx])
    pi = v1 / v1.sum()
    return pi

def entropy_rate(W, err=1e-8):
    pi = get_pi(W)
    H_rate = 0
    for i, P in enumerate(W):
        H_rate += - np.sum(pi[i] * np.sum(P * np.log2(P+err)))
        # H_rate += - np.sum(P * np.log2(P+err))
    return H_rate