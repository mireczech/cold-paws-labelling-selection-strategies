import numpy as np
from pdb import set_trace as pb
import time

from threadpoolctl import threadpool_info, threadpool_limits
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity

# pprint(threadpool_info())

def euclidean_distances_streamlined(X, Y, n_jobs=1, metric='euclidean'):
    with threadpool_limits(limits=n_jobs):
        if metric == 'euclidean':
            # XX = row_norms(X, squared=True)[:, np.newaxis]
            # YY = row_norms(Y, squared=True)[np.newaxis, :]

            XX = np.einsum("ij,ij->i", X, X)[:, np.newaxis]
            YY = np.einsum("ij,ij->i", Y, Y)[np.newaxis, :]

            # distances = -2 * safe_sparse_dot(X, Y.T, dense_output=True)
            distances = -2 * np.dot(X, Y.T)

            distances += XX
            distances += YY
            np.maximum(distances, 0, out=distances)
            if X is Y:
                np.fill_diagonal(distances, 0)

            return np.sqrt(distances, out=distances)
        elif metric == 'cosine_similarity':
            distances = 1-cosine_similarity(X, Y)
            return distances
