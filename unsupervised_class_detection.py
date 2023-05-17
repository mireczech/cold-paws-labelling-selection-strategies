import argparse
import os
import shutil

import copy

from itertools import product
import pandas as pd
import numpy as np
from pdb import set_trace as pb

import sys

from os.path import exists
import glob
import os
import re
import yaml
from argparse import Namespace

import uuid

# unsupervised class discovery
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--processed_data", type=str, required=True)
    args = parser.parse_args()
    return args

def import_config(args):
    with open(args.config, 'r') as file:
        new_args = yaml.safe_load(file)
        new_args = Namespace(**new_args)
    new_args.run_name = args.config
    return new_args

if __name__ == "__main__":
    args_parsed = parse_args()
    processed_data = args_parsed.processed_data #['data_processed/cifar10-resnet18sk0-1024-normed-bph.pickle']
    args = import_config(args_parsed)

    run_id = uuid.uuid4().hex[:10]

    args.config_path = "run/"
    args.PBS_path = "staging/"

    hypercomb = pd.DataFrame(list(
        product(args.budget, args.method
            )
        ), columns=[
            'budget', 'method'
        ])

    i = 1

    indices_list = []
    data_list = []

    for row in hypercomb.to_dict(orient="records"):
        config = copy.deepcopy(args)

        config.data = processed_data
        config.budget = row['budget']
        config.method = row['method']

        # config.data = config.data.replace('.', '_')
        config.data = config.data.replace('/', '_')

        output_folder = 'output/results_'+config.data+'_'+config.method+'_'+str(config.budget)+'_'+str(config.start)

        for i in range(config.reps):
            try:
                indices = pd.read_csv(output_folder + '/X' + str(config.budget) + '-' + str(i) + '-indicies.csv')
                data = pd.read_csv(output_folder + '/data' + str(config.budget) + '-' + str(i) + '-indicies.csv')

                indices['budget'] = row['budget']
                indices['method'] = row['method']
                indices['data_file'] = processed_data
                indices['seed'] = i
                indices['transform'] = data['transform'].values[0]
                if 'labels' in indices:
                    data['unique_classes'] = np.unique(indices['labels']).shape[0]

                indices_list.append(indices)
                data_list.append(data)
            except:
                print('failed')
                print(output_folder + '/X' + str(config.budget) + '-' + str(i) + '-indicies.csv')

    indices_all = pd.concat(indices_list)
    data_all = pd.concat(data_list)

    file_name = 'output/results_'+config.data
    file_name = file_name.replace('/', '_')

    indices_all.to_csv('data_clusters/'+file_name + '_indices_df.csv')
    data_all.to_csv('data_clusters/'+file_name + '_summary_df.csv')
