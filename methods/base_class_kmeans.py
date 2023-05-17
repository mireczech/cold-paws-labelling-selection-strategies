import numpy as np
import numpy
from pdb import set_trace as pb
import pandas as pd

from .base_class_stats import ExperimentStats

from tqdm import tqdm

import torch
import copy
from sklearn_extra.cluster import KMedoids

class ExperimentStatsKMeans(ExperimentStats):

    def kmediods(self, budget=100, seed=0, metric='euclidean'):
        print('kmediods')
        self.metric = metric
        if metric == 'cosine_similarity':
            metric = 'cosine'
        self.budget = budget
        kmedoids_fitted = KMedoids(n_clusters=budget, random_state=seed, init='k-medoids++', metric=metric).fit(self.data)
        self.indices = kmedoids_fitted.medoid_indices_

