
from methods.base_class import ExperimentImutable
from pdb import set_trace as pb
import pandas as pd
import numpy as np
import pickle
import argparse

# # ==============================================

experiment_data = ExperimentImutable(path='data/bm33708.csv', seed=0, n=-1, dims=-1, drop_first=True, sep=' ')
file = open('data_processed/'+'bm33708'+'.pickle', 'wb')
pickle.dump(experiment_data, file)
file.close()

experiment_data = ExperimentImutable(path='data/ch71009.csv', seed=0, n=-1, dims=-1, drop_first=True, sep=' ')
file = open('data_processed/'+'ch71009'+'.pickle', 'wb')
pickle.dump(experiment_data, file)
file.close()

experiment_data = ExperimentImutable(path='data/sw24708.csv', seed=0, n=-1, dims=-1, drop_first=True, sep=' ')
file = open('data_processed/'+'sw24708'+'.pickle', 'wb')
pickle.dump(experiment_data, file)
file.close()

experiment_data = ExperimentImutable(path='data/cifar10_simclr_encodings_bph.csv', 
		path_labels = 'data/cifar10_simclr_encodings_labels.csv',
		seed=0, n=-1, dims=-1, drop_first=False, sep=',',
		transform='t-SNE', transform_nrep=20)
file = open('data_processed/'+'cifar10'+'.pickle', 'wb')
pickle.dump(experiment_data, file)
file.close()

experiment_data = ExperimentImutable(path='data/imagenette_simclr_encodings_bph.csv', 
		path_labels = 'data/imagenette_simclr_encodings_labels.csv',
		seed=0, n=-1, dims=-1, drop_first=False, sep=',',
		transform='t-SNE', transform_nrep=20)
file = open('data_processed/'+'imagenette'+'.pickle', 'wb')
pickle.dump(experiment_data, file)
file.close()
