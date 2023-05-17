from main_func import *
from pdb import set_trace as pb
import argparse
from argparse import Namespace
import os
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config/test.yaml', type=str, help='foo help')
parser.add_argument('--dataset', default='config/test.yaml', type=str, help='foo help')
args = parser.parse_args()
data = args.dataset

with open(args.config, 'r') as file:
	args = yaml.safe_load(file)
	args = Namespace(**args)

# ================================
seed = args.start
base_seed = args.start
# ================================

nrep = args.reps

print(data)
newPath = data.replace(os.sep, '_')
for budget in args.budget:
	for method in args.method:
		res = run_experiments_k_center(
			data_file = data,
			budget = budget,
			seed = seed,
			method = method, # greedy, compose
			output_folder = 'output/results_'+newPath+'_'+method+'_'+str(budget)+'_'+str(base_seed),
			nrep=nrep
			)

