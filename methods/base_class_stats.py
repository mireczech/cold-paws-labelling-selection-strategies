import numpy as np
import numpy
from pdb import set_trace as pb
import pandas as pd
from .distance_matrix import euclidean_distances_streamlined as pairwise_distances
import torch

class ExperimentStats:
    
    # =================================================
    def cluster_groups(self):
        groups = np.argmin(self.kcentres_matrix, axis=1)
        groups = self.indices[groups]
        return groups

    def center_distribution(self):
        dists = np.min(self.kcentres_matrix, axis=1)
        groups = self.cluster_groups()
        max_dists = []
        for i in range(self.budget):
            pick_group = groups == self.indices[i]
            group_index = np.arange(self.n)[pick_group]
            max_dist = np.max(dists[group_index])
            max_dists.append(max_dist)
        return max_dists

    # =================================================
    def center_distances(self):
        return self.kcentres_matrix[self.indices, :]
    # =================================================
    def eval_distances(self):
        # np.sum(self.kcentres_matrix[np.array(self.indices), :] < self.kcentres_matrix.min(axis=1).max())
        return np.max(np.min(self.kcentres_matrix, axis=1))

    def eval_distance_between_centers(self):
        center_dists = self.center_distances()
        np.fill_diagonal(center_dists, np.Inf)
        # np.sum(self.kcentres_matrix[np.array(self.indices), :] < self.kcentres_matrix.min(axis=1).max())
        return np.min(center_dists, axis=1)

    # =================================================
    def eval_distances_test(self, test_data):
        data_centers = self.data[self.indices, :]
        test_dist = pairwise_distances(test_data, data_centers)
        return np.max(np.min(test_dist, axis=1))

    # =================================================
    def mean_distances(self):
        # np.sum(self.kcentres_matrix[np.array(self.indices), :] < self.kcentres_matrix.min(axis=1).max())
        return np.mean(np.min(self.kcentres_matrix, axis=1))

    def mean_distances_test(self, test_data):
        data_centers = self.data[self.indices, :]
        test_dist = pairwise_distances(test_data, data_centers)
        return np.mean(np.min(test_dist, axis=1))

    # =================================================
    def center_overlap(self, n_overlap=1):
        dists = self.kcentres_matrix
        dists = dists <= self.dist
        dists = np.sum(dists, axis=1)

        groups = self.cluster_groups()
        overlaps = []
        for i in range(self.budget):
            pick_group = groups == self.indices[i]
            group_index = np.where(pick_group)[0]
            overlap = np.mean(dists[group_index] > n_overlap)
            overlaps.append(overlap)
        overlaps = np.array(overlaps)
        return overlaps

