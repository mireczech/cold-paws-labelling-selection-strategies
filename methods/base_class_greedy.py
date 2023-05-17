import numpy as np
import numpy
from pdb import set_trace as pb
import pandas as pd
import copy
from .distance_matrix import euclidean_distances_streamlined as pairwise_distances

from .base_class_stats import ExperimentStats
from .base_class_kcentres import ExperimentStatsKCentres
from .base_class_kmeans import ExperimentStatsKMeans
from .base_transform import TSNE_transform
import torchvision
import os

class ExperimentStatsGreedy(ExperimentStatsKCentres, ExperimentStatsKMeans):
    def __init__(self, experiment_data, labels=None, index=None, transform='None', seed=0, metric='euclidean'):
        self.starting_index = None
        self.silent = False

        if transform != 'None':
            if metric == 'euclidean':
                self.data = experiment_data.data_transformed_euclidean[seed % len(experiment_data.data_transformed_euclidean)]
            elif metric == 'cosine':
                self.data = experiment_data.data_transformed_cosine[seed % len(experiment_data.data_transformed_cosine)]
        else:
            self.data = experiment_data.data

        if index is None:
            self.data_idx = np.arange(self.data.shape[0])
            self.labels = labels
        else:
            self.data = self.data[index]
            self.data_idx = index
            self.labels = labels[index]

    # =================================================

    def return_indices(self):
        return self.data_idx[self.indices]

    def return_labels(self):
        return self.labels[self.indices]

    # =================================================

    def return_stats(self):
        self.kcentres_matrix = pairwise_distances(self.data, self.data[self.indices, :], metric='euclidean')
        mini_max_dist = self.eval_distances()
        maxi_min_dist = np.min(self.eval_distance_between_centers())
        kmediods_dist = np.sum(np.min(self.kcentres_matrix, axis=1))

        self.kcentres_matrix = pairwise_distances(self.data, self.data[self.indices, :], metric='cosine_similarity')
        mini_max_dist_csd = self.eval_distances()
        maxi_min_dist_csd = np.min(self.eval_distance_between_centers())
        kmediods_dist_csd = np.sum(np.min(self.kcentres_matrix, axis=1))

        return {
                'maxi_min': maxi_min_dist, 'mini_max': mini_max_dist, 'kmediods': kmediods_dist,
                'maxi_min_csd': maxi_min_dist_csd, 'mini_max_csd': mini_max_dist_csd, 'kmediods_csd': kmediods_dist_csd
                }