
from methods.base_class_greedy import ExperimentStatsGreedy
from methods.base_transform import *
from pdb import set_trace as pb
import pickle
import pandas as pd
import os
import multiprocessing as mp
import numpy as np

import time
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from functools import partial
import re
import copy

import importlib

def write_exp_output(args, iteration=0, base_seed=0):
    cifar10_data, data_file, method, transform, args_method, output_folder, arguments = args
    arguments['seed'] = base_seed+iteration

    if hasattr(cifar10_data, 'labels'):
        labels = cifar10_data.labels
    else:
        labels = None

    if hasattr(cifar10_data, 'subset_index'):
        subset_index = cifar10_data.subset_index
    else:
        subset_index = None

    if 'metric' in arguments.keys():
        if arguments['metric'] == 'cosine_similarity':
            metric = 'cosine'
        else:
            metric = 'euclidean'
    else:
        metric = 'euclidean'

    cifar10 = ExperimentStatsGreedy(cifar10_data, labels=labels, index=subset_index, transform=transform, seed=arguments['seed'], metric=metric)

    cifar10.output_folder = output_folder
    cifar10.silent = False

    arguments_apply = copy.deepcopy(arguments)
    if transform != 'None':
        arguments_apply['metric'] = 'euclidean'

    start = time.time()
    apply_func = getattr(cifar10, args_method)
    apply_func(**arguments_apply)
    end = time.time()
    total_time = end - start

    d = {'data_file':data_file, 'budget' : cifar10.budget, 'seed': arguments_apply['seed'], \
        'method': method, 'transform':transform,\
        'total_time': total_time}

    ind_df = pd.DataFrame(cifar10.return_indices(), columns=['indices'])
    if cifar10.labels is not None:
        ind_df['labels'] = cifar10.return_labels()

    stats = cifar10.return_stats()
    d.update(stats)
    stats = pd.DataFrame.from_dict(d, orient='index').T

    ind_df.to_csv(cifar10.output_folder + '/X'+str(cifar10.budget)+'-'+str(arguments_apply['seed'])+'-indicies.csv')
    stats.to_csv(cifar10.output_folder + '/data'+str(cifar10.budget)+'-'+str(arguments_apply['seed'])+'-indicies.csv')

    matplotlib_present = importlib.util.find_spec("matplotlib")
    if matplotlib_present is not None:
        if cifar10.labels is not None:
            labels = cifar10.labels
        else:
            labels = np.zeros(cifar10.data.shape[0])

        plot_transform(cifar10.data, ind_df['indices'].values, labels, cifar10.output_folder + '/P'+str(cifar10.budget)+'-'+str(arguments_apply['seed'])+'-indicies.png')
    if False:
        if transform != 'None':
            encodings = pd.DataFrame(cifar10.data).to_csv(cifar10.output_folder + '/encodings'+str(cifar10.budget)+'-'+str(arguments_apply['seed'])+'-indicies.csv')

    return d

def run_experiments_k_center(
                    data_file = 'data/cifar10-50000.pickle',
                    budget = 40,
                    seed = 0,
                    output_folder = '',
                    method = 'finetune', # greedy, compose
                    initial_k = 100,
                    nrep=1
                    ):

    start = time.time()
    if not os.path.exists(output_folder):
        try:
            os.mkdir(output_folder)
        except:
            pass

    ff = open(data_file, 'rb')
    cifar10_data = pickle.load(ff)
    ff.close()

    # https://stackoverflow.com/questions/7894791/use-numpy-array-in-shared-memory-for-multiprocessing

    if 'tsne_' in method:
        transform = 'TSNE'
        method = re.sub("tsne_", "", method)
    else:
        transform = 'None'
        method = method


    if method == 'finetune':
        args_dict = {'seed':seed, 'k_budget': budget, 'finetune_iters': 100, 'redundant_thresh': 0.9999, 'starting_index': None}
        args_method = 'base_center_compose'
    elif method == 'finetune_all_at_once':
        args_dict = {'seed':seed, 'k_budget': budget, 'finetune_iters': 100, 'redundant_thresh': 0.9999, 'starting_index': None}
        args_method = 'base_center_compose_allatonce'
    elif method == 'finetune_1center':
        args_dict = {'seed':seed, 'k_budget': budget, 'finetune_iters': 100, 'redundant_thresh': 0.9999}
        args_method = 'base_center_compose_start1center'
    elif method == 'greedy':
        args_dict = {'seed':seed, 'budget': budget}
        args_method = 'greedy_k_center'
    elif method == 'greedy_stratified':
        args_dict = {'seed':seed, 'budget': budget}
        args_method = 'greedy_k_center_stratified'
    elif method == 'greedy_1class':
        args_dict = {'seed':seed, 'budget': budget, 'indices':[-1]}
        args_method = 'greedy_k_center'
    elif method == 'compose':
        args_dict = {'seed':seed, 'k_budget': budget, 'finetune_iters': 100, 'redundant_thresh': 0.9999, 'initial_k': initial_k}
        args_method = 'base_k_center_compose'
    elif method == 'random':
        args_dict = {'seed':seed, 'budget': budget}
        args_method = 'random'
    elif method == 'random_1class':
        args_dict = {'seed':seed, 'budget': budget, 'indices':[-1]}
        args_method = 'random'
    elif method == 'random_labelled':
        args_dict = {'seed':seed, 'budget': budget}
        args_method = 'random_labelled'
    elif method == 'repulsive':
        args_dict = {'seed':seed, 'k_budget': budget, 'finetune_iters': 100, 'redundant_thresh': 0.9999, 'starting_index': None}
        args_method = 'repulsive_center_compose'
    elif method == 'repulsive_stratified':
        args_dict = {'seed':seed, 'k_budget': budget, 'finetune_iters': 100, 'redundant_thresh': 0.9999, 'starting_index': None}
        args_method = 'repulsive_stratified_center_compose'
    elif method == 'kmediods':
        args_dict = {'seed':seed, 'budget': budget}
        args_method = 'kmediods'
    elif method == 'greedy_csd':
        args_dict = {'seed':seed, 'budget': budget, 'metric': 'cosine_similarity'}
        args_method = 'greedy_k_center'
    elif method == 'finetune_csd':
        args_dict = {'seed':seed, 'k_budget': budget, 'finetune_iters': 100, 'redundant_thresh': 0.9999, 'starting_index': None, 'metric': 'cosine_similarity'}
        args_method = 'base_center_compose'
    elif method == 'repulsive_csd':
        args_dict = {'seed':seed, 'k_budget': budget, 'finetune_iters': 100, 'redundant_thresh': 0.9999, 'starting_index': None, 'metric': 'cosine_similarity'}
        args_method = 'repulsive_center_compose'
    elif method == 'kmediods_csd':
        args_dict = {'seed':seed, 'budget': budget, 'metric': 'cosine_similarity'}
        args_method = 'kmediods'
   
    write_exp = partial(write_exp_output)
    args = (cifar10_data, data_file, method, transform, args_method, output_folder, args_dict)
    
    for i in range(nrep):
        d = write_exp(args, iteration=i, base_seed=seed)

    end = time.time()
    print(end - start)

            
