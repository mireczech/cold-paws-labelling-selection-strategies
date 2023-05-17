
import numpy as np
from .distance_matrix import euclidean_distances_streamlined as pairwise_distances
from pdb import set_trace as pb
from .kcentre_greedy import coreset_k_centres_greedy

def distance_matrix(data):
    dist_mat = pairwise_distances(data, data, n_jobs=1)
    return dist_mat

def finetune_exact(data_u, kcentures_u):
    dist_mat_u = distance_matrix(data_u)
    # dist_mat_u = self.dist_mat[tuple(np.meshgrid(group_index, group_index))]
    dist_mat_u = np.max(dist_mat_u, axis=1)
    return dist_mat_u

def finetune_intersection(data_u, kcentures_u, check_against_exact=False):
    closet_indices = np.argmin(kcentures_u.T, axis=1)
    furthest_indices = np.argmax(kcentures_u.T, axis=1)

    poor_hull_indices = np.unique(np.concatenate((closet_indices, furthest_indices)))
    dist_mat_convex = pairwise_distances(data_u, data_u[poor_hull_indices, :], n_jobs=1)
    i = 0
    while True:
        i = i + 1
        if i > 100:
            pb()
        min_candidate = np.argmin(np.max(dist_mat_convex, axis=1))

        dist_mat_new = pairwise_distances(data_u[min_candidate:min_candidate+1, :],  data_u, n_jobs=1)
        furthest_dist = np.argmax(dist_mat_new)

        if furthest_dist in poor_hull_indices:
            dist_mat_u = np.max(dist_mat_convex, axis=1)
            break
        else:
            poor_hull_indices = np.append(poor_hull_indices, furthest_dist)
            dist_mat_furthest = pairwise_distances(data_u[furthest_dist:furthest_dist+1, :],  data_u, n_jobs=1)
            dist_mat_convex = np.hstack((dist_mat_convex, dist_mat_furthest.T))


    # dist_mat_u

    if check_against_exact:
        res = finetune_exact(data_u, kcentures_u)
        if np.argmin(res) != np.argmin(dist_mat_u):
            pb()
    return dist_mat_u

