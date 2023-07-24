import numpy as np
import numpy
from pdb import set_trace as pb
from tqdm import tqdm
import pandas as pd

from .distance_matrix import euclidean_distances_streamlined as pairwise_distances

from .base_transform import TSNE_transform

import torch
import importlib

class ExperimentImutable:
    def __init__(self, path='', path_labels=None, subset_index = None, test_path = '', test_labels=None, n=-1, dims=-1, drop_first=False, sep=',', 
            seed=0, transform='None', transform_nrep = 20):
        self.path = path
        self.seed = seed
        self.n = n

        print('creating embeddings')
        numpy.random.seed(self.seed)
        if path != '':
            data, labels = self.initialise_embeddings(path, seed, n, drop_first, sep, path_labels)
        else:
            data, labels = self.initialise_gaussian(seed, n, dims)

        self.data = data
        self.labels = labels
        self.dims = self.data.shape[1]

        if subset_index is not None:
            self.subset_index = subset_index

        if test_path != '':
            data, labels = self.initialise_embeddings(test_path, seed, n, drop_first, sep, test_labels)
            self.test_data = data
            self.test_labels = labels
        else:
            self.test_data = None
            self.test_labels = None

        print('transforming')
        if transform != 'None':
            # self.data_transformed_euclidean = []
            self.data_transformed_cosine = []
            for i in tqdm(range(transform_nrep)):
                # self.data_transformed_euclidean.append(TSNE_transform(self.data, seed=seed+i, metric='euclidean'))
                self.data_transformed_cosine.append(TSNE_transform(self.data, seed=seed+i, metric='cosine'))

    def plot_clusters(self, save_name='', nrep=2):
        from .base_transform import plot_transform
        for i in range(nrep): # len(self.data_transformed_euclidean)
            # plot_transform(self.data_transformed_euclidean[i], None, self.labels, 'data_clusters/' + save_name +'-euclidean'+str(i)+'.png')
            plot_transform(self.data_transformed_cosine[i], None, self.labels, 'data_clusters/' + save_name +'-cosine'+str(i)+'.png')

    def initialise_embeddings(self, path,seed, n, drop_first, sep, path_labels = None):
        # data = np.genfromtxt(path)
        data = pd.read_csv(path,header=0, sep=sep)
        data = data.values
        data = np.float64(data)
        if (drop_first):
            data = data[:, 1:]

        if path_labels is not None:
            labels = pd.read_csv(path_labels,header=0, sep=sep)
            labels = labels.values.squeeze()
        else:
            labels = None

            
        # np.unique(data.round(decimals=6), axis = 0)
        # data = data[0:10000, :]
        if n != -1:
            indices = np.random.choice(data.shape[0], n, replace=False)
            data = data[indices, :]
            if path_labels is not None:
                labels = labels[indices]
        return data, labels

    # =================================================

    def initialise_gaussian(self, seed, n, dims):
        data = numpy.random.normal(loc=0, scale=1, size=(n, dims))
        return data, None

    # =================================================
    def distance_matrix(self, data):
        dist_mat = pairwise_distances(data, data, n_jobs=1)
        return dist_mat

