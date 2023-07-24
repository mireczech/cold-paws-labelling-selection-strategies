#!/bin/bash

python main_results.py --config config/our.yaml --dataset data_processed/our_matek.pickle
python main_results.py --config config/our.yaml --dataset data_processed/our_isic.pickle
python main_results.py --config config/our.yaml --dataset data_processed/our_retinopathy.pickle
python main_results.py --config config/our.yaml --dataset data_processed/our_jurkat.pickle
python main_results.py --config config/our.yaml --dataset data_processed/our_cifar10.pickle
