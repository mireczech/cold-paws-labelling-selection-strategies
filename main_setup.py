
from methods.base_class import ExperimentImutable
from pdb import set_trace as pb
import pandas as pd
import numpy as np
import pickle
import argparse
import os
import re
import time
import json
import tqdm

# folder definition
data_dir = 'data'
data_processed_dir = 'data_processed'

# searching folder
dataset_models = list([
    re.search(r'our_[^_]+_(.+)\.csv$', path).group(1)
    for path in os.listdir(data_dir)
    if path.startswith('our_features_')
])

# # TODO: debug
# dataset_models = ['matek_simclr_v1']

print(f'dataset/models found: {dataset_models}')

# our data
for dataset_model in tqdm.tqdm(dataset_models):
	print(f'processing {dataset_model}')

	# transform_nrep=20
	# transform_nrep=1
	transform_nrep=5

	experiment_data = ExperimentImutable(
		path=os.path.join(data_dir, f'our_features_{dataset_model}.csv'), 
		path_labels = os.path.join(data_dir, f'our_labels_{dataset_model}.csv'),
		seed=0, n=-1, dims=-1, drop_first=False, sep=',',
		transform='t-SNE', transform_nrep=transform_nrep
	)


	with open(os.path.join(data_processed_dir, f'{dataset_model}.pickle'), 'wb') as f:
		pickle.dump(experiment_data, f)
