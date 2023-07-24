
from methods.base_class import ExperimentImutable
from pdb import set_trace as pb
import pandas as pd
import numpy as np
import pickle
import argparse
import os

# # ==============================================

# experiment_data = ExperimentImutable(path='data/bm33708.csv', seed=0, n=-1, dims=-1, drop_first=True, sep=' ')
# file = open('data_processed/'+'bm33708'+'.pickle', 'wb')
# pickle.dump(experiment_data, file)
# file.close()

# experiment_data = ExperimentImutable(path='data/ch71009.csv', seed=0, n=-1, dims=-1, drop_first=True, sep=' ')
# file = open('data_processed/'+'ch71009'+'.pickle', 'wb')
# pickle.dump(experiment_data, file)
# file.close()

# experiment_data = ExperimentImutable(path='data/sw24708.csv', seed=0, n=-1, dims=-1, drop_first=True, sep=' ')
# file = open('data_processed/'+'sw24708'+'.pickle', 'wb')
# pickle.dump(experiment_data, file)
# file.close()

# experiment_data = ExperimentImutable(path='data/cifar10_simclr_encodings_bph.csv', 
# 		path_labels = 'data/cifar10_simclr_encodings_labels.csv',
# 		seed=0, n=-1, dims=-1, drop_first=False, sep=',',
# 		transform='t-SNE', transform_nrep=20)
# file = open('data_processed/'+'cifar10'+'.pickle', 'wb')
# pickle.dump(experiment_data, file)
# file.close()

# experiment_data = ExperimentImutable(path='data/imagenette_simclr_encodings_bph.csv', 
# 		path_labels = 'data/imagenette_simclr_encodings_labels.csv',
# 		seed=0, n=-1, dims=-1, drop_first=False, sep=',',
# 		transform='t-SNE', transform_nrep=20)
# file = open('data_processed/'+'imagenette'+'.pickle', 'wb')
# pickle.dump(experiment_data, file)
# file.close()

# our data
for dataset in [
	'matek',
    'isic',
    'retinopathy',
    'jurkat',
    'cifar10',
]:
	print(f'processing {dataset}')

	transform_nrep=20
	# transform_nrep=1
	experiment_data = ExperimentImutable(path=f'data/our_features_{dataset}.csv', 
			path_labels = f'data/our_labels_{dataset}.csv',
			seed=0, n=-1, dims=-1, drop_first=False, sep=',',
			transform='t-SNE', transform_nrep=transform_nrep)
	file = open(f'data_processed/our_{dataset}.pickle', 'wb')
	pickle.dump(experiment_data, file)
	file.close()
