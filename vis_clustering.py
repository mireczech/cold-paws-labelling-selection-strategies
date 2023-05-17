from methods.base_class import ExperimentImutable
from methods.base_class_greedy import ExperimentStatsGreedy
from pdb import set_trace as pb
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# https://builtin.com/data-science/tsne-python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from vis_clustering_helper import plot_clustering

# ========================

val_file = 'cifar10-resnet18sk0-1024-normed-bph'
label_order = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

labelled_points = 'output/results_data_processed_cifar10-resnet18sk0-1024-normed-bph.pickle_tsne_kmediods_csd_40_0/X40-0-indicies.csv'
labelled_points2 = 'output/results_data_processed_cifar10-resnet18sk0-1024-normed-bph.pickle_finetune_40_0/X40-0-indicies.csv'
plot_clustering(val_file, labelled_points, labelled_points2, label_order, name_suffix='tsne_all', tsne=False, encodings_metric='cosine_distance')

label_order = ['tench', 'english springer', 'cassette player', 'chain saw', 'church', 'french horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
val_file = 'imagenette-resnet18sk0-bignormed-rw-bph'

labelled_points = 'output/results_data_processed_imagenette-resnet18sk0-bignormed-rw-bph.pickle_tsne_kmediods_csd_40_0/X40-0-indicies.csv'
labelled_points2 = 'output/results_data_processed_imagenette-resnet18sk0-bignormed-rw-bph.pickle_finetune_40_0/X40-0-indicies.csv'
plot_clustering(val_file, labelled_points, labelled_points2, label_order, name_suffix='tsne_all', tsne=False, encodings_metric='cosine_distance')
