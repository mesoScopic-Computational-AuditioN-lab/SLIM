import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm

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
    return H_rate


def ada_lipo(f, domains, n_iter=100, n_proposals=200, verbose=False, return_k_hist=False, p=.1, max_retries=10):
    domains = np.array(domains, dtype=float)
    lows, highs = domains[:, 0], domains[:, 1]
    d = len(domains)

    if return_k_hist:
        k_hist = []

    def sample(n=1):
        points = np.random.uniform(lows, highs, size=(n, d))
        return points[0] if n == 1 else points

    def U(proposals, X, y, k):
        dists = np.linalg.norm(proposals[:, None] - X[None], axis=2)
        return np.min(y[None, :] + k * dists, axis=1)

    def update_k(k, x_new, y_new, X, y):
        if len(X) == 0:
            return k
        dists = np.linalg.norm(X - x_new, axis=1)
        mask = dists > 0
        if not np.any(mask):
            return k
        slopes = np.abs(y[mask] - y_new) / dists[mask]
        return max(k, slopes.max())

    X_list = [sample(), sample()]
    y_list = [f(X_list[0]), f(X_list[1])]

    k = update_k(0.0, X_list[1], y_list[1], np.array([X_list[0]]), np.array([y_list[0]]))
    if return_k_hist:
        k_hist.append(k)

    for _ in tqdm(range(n_iter), disable=not verbose):
        X_arr, y_arr = np.array(X_list), np.array(y_list)
        M_t = y_arr.max()

        x_next = None
        if k == 0 or np.random.rand() < p:
            x_next = sample()
        else:
            for _ in range(max_retries):
                proposals = sample(n_proposals)
                optimistic = U(proposals, X_arr, y_arr, k) >= M_t
                if np.any(optimistic):
                    promising = proposals[optimistic]
                    x_next = promising[np.random.randint(len(promising))]
                    break
            if x_next is None:
                x_next = sample()

        y_next = f(x_next)
        k = update_k(k, x_next, y_next, X_arr, y_arr)

        X_list.append(x_next)
        y_list.append(y_next)
        if return_k_hist:
            k_hist.append(k)

    best = np.argmax(y_list)
    if return_k_hist:
        return X_list[best], y_list[best], k_hist
    return X_list[best], y_list[best]

def fit_slim_star(betas, labels, model, W_init):
    from sklearn.preprocessing import StandardScaler
    from .utils import ada_lipo

    betas = StandardScaler().fit_transform(betas.reshape(-1,1)).ravel()

    def f(params, return_score=False):
        lr = params
        model.W = W_init
        model.flush_history()
        model.fit(labels, learning_rate=lr)
        surprizes = StandardScaler().fit_transform(np.array(model.surprise_hist).reshape(-1,1)).ravel()
        surprizes = surprizes[1::2]
        return -np.mean((betas - surprizes) ** 2)

    theta_best, mse_best = ada_lipo(f=f,
                                    domains=[(0., 1.)],
                                    return_k_hist=False,
                                    n_iter=50,
                                    n_proposals=100,)

    # run on best
    model.flush_history()
    model.fit(labels, learning_rate=theta_best[0])
    surprizes = StandardScaler().fit_transform(np.array(model.surprise_hist).reshape(-1,1)).ravel()
    surprizes = surprizes[1::2]
    return surprizes, mse_best # k_hist


def fit_slim_stars(data, dim, labels, pretrain=None, labels_map=None, n_jobs=-1):
    from joblib import Parallel, delayed
    from .model import SLIM

    print(f'I\'m fitting {data.shape[0]} voxels of {len(labels)} trials...')
    if isinstance(labels[0], str):
        # assume labels are represented as <key>.<value>
        labels = np.concatenate([label.split('.') for label in labels])
        print(f'detected str labels, I\'ll encode them with integers, of dim={len(np.unique(labels))}')
        labels_map = {label:idx for idx, label in enumerate(np.unique(labels))}
        labels = list(map(labels_map.get, labels))

    model = SLIM(dim=dim, sensor=np.eye(dim, dtype=float))
    # pretrain / initialize
    pretrain = list(map(labels_map.get, pretrain))
    model.fit(pretrain, learning_rate=.2)
    model.flush_history()
    W_init = model.W

    def fit_wrapper(data_item, model, W_init):
        return fit_slim_star(data_item, labels, dim, pretrain, model, W_init)

    completed = Parallel(n_jobs=n_jobs,
                         backend='loky',
                         return_as='generator',
                         inner_max_num_threads=1)(
        delayed(fit_wrapper)(x, model, W_init) for x in data
    )
    results = list(tqdm(completed, total=len(data)))

    return results

