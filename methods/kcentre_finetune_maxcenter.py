
import numpy as np
from .distance_matrix import euclidean_distances_streamlined as pairwise_distances
from pdb import set_trace as pb
from .kcentre_greedy import coreset_k_centres_greedy

def distance_matrix(data):
    dist_mat = pairwise_distances(data, data, n_jobs=1)
    return dist_mat

def finetune_repulsion(data_u, kcentures_u):
    kcentures_u = kcentures_u
    dist_mat_u = np.min(kcentures_u, axis=1)
    return dist_mat_u
