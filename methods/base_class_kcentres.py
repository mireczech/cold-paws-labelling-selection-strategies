import numpy as np
import numpy
from pdb import set_trace as pb
import pandas as pd

from .base_class_stats import ExperimentStats
from .kcentre_greedy import coreset_k_centres_greedy
from .kcentre_finetune import *
from .kcentre_finetune_maxcenter import *
from .distance_matrix import euclidean_distances_streamlined as pairwise_distances

from tqdm import tqdm

import torch
import copy


class ExperimentStatsKCentres(ExperimentStats):

    # =================================================

    def random(self, budget=100, seed=0, indices=None):
        self.budget = budget

        if indices is not None and indices[0] == -1:
            n_labels = numpy.unique(self.labels)
            indices = self._random_labelled(budget=budget, seed=seed)
            budget = budget - n_labels.shape[0]

        numpy.random.seed(seed)
        index = np.random.permutation(self.data.shape[0])

        if indices is not None and indices[0] == -1:
            self.indices = np.hstack((index[0:budget], indices))
        else:
            self.indices = index[0:budget]
            
        self.starting_index = self.indices[0:1]
        # np.unique(self.labels[self.indices], return_counts=True)

        # self.kcentres_matrix = pairwise_distances(self.data, self.data[self.indices, :])
        # self.dist = self.eval_distances()


    def _random_labelled(self, budget=100, seed=0):
        n_labels = numpy.unique(self.labels)
        per_label_budget = budget//n_labels.shape[0]

        numpy.random.seed(seed)
        indices_list = []
        for i in range(n_labels.shape[0]):
            label = np.nonzero(self.labels == i)[0]
            subset = np.random.permutation(label.shape[0])
            subset = subset[0:per_label_budget]
            indices_list.append(label[subset])
        indices = np.hstack(indices_list)
        return indices

    def random_labelled(self, budget=100, seed=0):
        self.budget = budget

        self.indices = self._random_labelled(budget=budget, seed=seed)
        self.starting_index = self.indices[0:1]

        # self.kcentres_matrix = pairwise_distances(self.data, self.data[self.indices, :])
        # np.unique(self.labels[self.indices], return_counts=True)
        # self.dist = self.eval_distances()


    # =================================================

    def _greedy_k_center(self, data, labels, budget=100, seed=0, indices=None, save_matrix=True):
        if indices is None:
            numpy.random.seed(0)
            index = np.random.permutation(data.shape[0])
            indices = np.array([index[seed]])
        elif indices[0] == -1:
            n_labels = numpy.unique(labels)
            indices = self._random_labelled(budget=budget, seed=seed)

        greedy_indices, dist_mat, mat_min, mat_min_center = coreset_k_centres_greedy(
                            data, budget, indices=indices, 
                            silent=self.silent, save_matrix=save_matrix, metric=self.metric)
        return greedy_indices, indices, dist_mat, mat_min, mat_min_center


    def greedy_k_center(self, budget=100, seed=0, indices=None, save_matrix=True, metric='euclidean'):
        self.budget = budget
        self.metric = metric
        greedy_indices, indices, dist_mat, mat_min, mat_min_center = \
            self._greedy_k_center(self.data, self.labels, budget=self.budget, seed=seed, indices=indices, save_matrix=save_matrix)

        if self.starting_index is None:
            self.starting_index = copy.deepcopy(indices)

        self.indices = greedy_indices

        self.kcentres_matrix = dist_mat
        self.kcentres_min = mat_min
        self.kcentres_min_center = mat_min_center

        self.dist = np.max(mat_min)

    def greedy_k_center_stratified(self, budget=100, seed=0, indices=None, save_matrix=True, metric='euclidean'):
        self.budget = budget
        self.metric = metric

        classes = np.unique(self.labels).shape[0]
        class_budget = budget//classes
        i = 0
        greedy_indices, indices, dist_mat, mat_min, mat_min_center = ([], [], [] ,[], [])
        total_indices = np.arange(self.data.shape[0])

        for clss in range(classes):
            i += 1
            data = self.data[self.labels == clss, :]
            labels = self.labels[self.labels == clss]
            map_indices = total_indices[self.labels == clss]

            _greedy_indices, _indices, _dist_mat, _mat_min, _mat_min_center = \
                self._greedy_k_center(data, labels, budget=class_budget, seed=seed+i, indices=None, save_matrix=save_matrix)
            greedy_indices.append(map_indices[_greedy_indices])
            indices.append(map_indices[_greedy_indices])
            dist_mat.append(_dist_mat)
            mat_min.append(_mat_min)
            mat_min_center.append(_mat_min_center)


        if self.starting_index is None:
            self.starting_index = copy.deepcopy(np.concatenate(indices))
        # These are wrong but I don't care
        self.indices = np.concatenate(greedy_indices)

        self.kcentres_matrix = pairwise_distances(self.data, self.data[self.indices, :], metric=self.metric)
        self.kcentres_min = np.concatenate(mat_min)
        self.kcentres_min_center = np.concatenate(mat_min_center)

        self.dist = np.max(self.kcentres_min)

    # =================================================
    # https://stackoverflow.com/questions/27414060/find-the-diameter-of-a-set-of-n-points-in-d-dimensional-space
    # https://stackoverflow.com/questions/2736290/how-to-find-two-most-distant-points
    def find_1center(self, data_u, kcentures_u,  method='intersection'):
        if method == 'exact':
            dist_mat_u = finetune_exact(data_u, kcentures_u)
        elif method == 'intersection':
            dist_mat_u = finetune_intersection(data_u, kcentures_u)
        elif method == 'repulsion':
            dist_mat_u = finetune_repulsion(data_u, kcentures_u)

        if method == 'repulsion':
            best_center = np.argmax(dist_mat_u)
        else:
            best_center = np.argmin(dist_mat_u)
        return best_center

    # =================================================
    def start_1center(self, method='intersection'):

        data_u = self.data

        max_dims = np.argmax(data_u, axis=0)
        min_dims = np.argmin(data_u, axis=0)

        minmax_dims = np.concatenate((max_dims, min_dims))
        minmax_dims = np.unique(minmax_dims)
        kcentures_u = pairwise_distances(data_u, data_u[minmax_dims, :], metric=self.metric)

        best_center = self.find_1center(data_u, kcentures_u,  method=method)
        return best_center


    # =================================================
    def center_finetune(self, method='intersection', del_center = False, ):
        groups = self.cluster_groups()

        # ====================================
        new_centers = self.indices
        for i in range(self.budget):
            pick_group = groups == self.indices[i]

            group_index = np.where(pick_group)[0]
            data_u = self.data[group_index, :]
            kcentures_u = self.kcentres_matrix[group_index, :]
            if del_center:
                kcentures_u = np.delete(kcentures_u, i, 1)

            best_center = self.find_1center(data_u, kcentures_u,  method=method)
            best_center_index = group_index[best_center]
            new_centers[i] = best_center_index
            if del_center:
                # pb()
                self.kcentres_matrix[:, i:i+1] = pairwise_distances(self.data, self.data[best_center_index:best_center_index+1, :], metric=self.metric)

        new_centers = np.array(new_centers)

        self.indices = new_centers
        self.kcentres_matrix = pairwise_distances(self.data, self.data[self.indices, :], metric=self.metric)
        
        if method == 'exact' or method == 'intersection':
            self.dist = self.eval_distances()
        elif method == 'repulsion':
            self.dist = np.min(self.eval_distance_between_centers())

    # =================================================
    def center_remove_redundant(self, threshold=0.9999):
        center_overlap = self.center_overlap(n_overlap=1)

        nonredundant_points = np.argwhere(center_overlap < threshold)
        pruned_indices = self.indices[nonredundant_points]
        self.greedy_k_center(budget=self.budget, seed=0, indices=pruned_indices)

    # =================================================
    def center_finetune_multi(self, iterations=100, **kwargs):
        dists = []
        dists.append(self.dist)
        for i in tqdm(range(iterations), ncols=100, disable=self.silent):
            self.center_finetune(**kwargs)
            dists.append(self.dist)
            if dists[-1] == dists[-2]:
                break
        return dists


    # =================================================
    def center_finetune_redundant_multi(self, iterations=100, threshold=0.9999):
        dists = []
        dists.append(self.dist)
        for i in tqdm(range(iterations), ncols=100, disable=self.silent):
            self.center_finetune()
            self.center_remove_redundant(threshold = threshold)
            dists.append(self.dist)
            if dists[-1] == dists[-2]:
                break
        return dists

    # =================================================
    def base_center_compose(self, seed=0, k_budget=100, finetune_iters=100, redundant_thresh=0.9999, starting_index=None, metric='euclidean'):
        self.greedy_k_center(budget=k_budget, seed=seed, indices=starting_index, metric=metric)
        self.center_finetune_multi(iterations=finetune_iters)
        self.center_remove_redundant(threshold = redundant_thresh)
        self.center_finetune_multi(iterations=finetune_iters)

    # =================================================
    def base_center_compose_start1center(self, seed=0, k_budget=100, finetune_iters=100, redundant_thresh=0.9999, metric='euclidean'):
        self.metric = metric
        starting_index = np.array([self.start_1center(method='intersection')])
        self.greedy_k_center(budget=k_budget, seed=seed, indices=starting_index, metric=metric)
        self.center_finetune_multi(iterations=finetune_iters)
        self.center_remove_redundant(threshold = redundant_thresh)
        self.center_finetune_multi(iterations=finetune_iters)


    # =================================================
    def base_center_compose_allatonce(self, seed=0, k_budget=100, finetune_iters=100, redundant_thresh=0.9999, starting_index=None, metric='euclidean'):
        self.greedy_k_center(budget=k_budget, seed=seed, indices=starting_index, metric=metric)
        self.center_finetune_redundant_multi(iterations=finetune_iters, threshold = redundant_thresh)
    # =================================================

    def base_k_center_compose(self, seed=0, initial_k=100, k_budget=100, finetune_iters=100, redundant_thresh=0.9999, metric='euclidean'):

        self.base_center_compose(seed=seed, k_budget=initial_k, finetune_iters=finetune_iters, redundant_thresh=redundant_thresh, starting_index=None, metric=metric)

        initial_k_indices = self.indices

        solution_dists = []
        for i in range(initial_k):
            self.base_center_compose(seed=seed, k_budget=k_budget, finetune_iters=finetune_iters, redundant_thresh=redundant_thresh, starting_index=np.array([initial_k_indices[i]]))
            solution_dists.append(self.dist)

        self.dist = np.min(np.array(solution_dists))

        return initial_k_indices, solution_dists

    # =================================================
    def repulsive_center_compose(self, seed=0, k_budget=100, finetune_iters=100, redundant_thresh=0.9999, starting_index=None, metric='euclidean'):
        self.greedy_k_center(budget=k_budget, seed=seed, indices=starting_index, metric=metric)
        self.center_finetune_multi(iterations=finetune_iters, method='repulsion', del_center = True)

    def repulsive_stratified_center_compose(self, seed=0, k_budget=100, finetune_iters=100, redundant_thresh=0.9999, starting_index=None, metric='euclidean'):
        self.greedy_k_center_stratified(budget=k_budget, seed=seed, metric=metric)
        self.center_finetune_multi(iterations=finetune_iters, method='repulsion', del_center = True)
