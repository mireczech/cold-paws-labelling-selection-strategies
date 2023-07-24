#!/bin/bash

python main_setup.py

# simclr
python main_results.py --config config/our.yaml --dataset data_processed/matek_simclr_v1.pickle
python main_results.py --config config/our.yaml --dataset data_processed/isic_simclr_v1.pickle
python main_results.py --config config/our.yaml --dataset data_processed/retinopathy_simclr_v1.pickle
python main_results.py --config config/our.yaml --dataset data_processed/jurkat_simclr_v1.pickle
python main_results.py --config config/our.yaml --dataset data_processed/cifar10_simclr_v1.pickle

# swav
python main_results.py --config config/our.yaml --dataset data_processed/matek_swav_v1.pickle
python main_results.py --config config/our.yaml --dataset data_processed/isic_swav_v1.pickle
python main_results.py --config config/our.yaml --dataset data_processed/retinopathy_swav_v1.pickle
python main_results.py --config config/our.yaml --dataset data_processed/jurkat_swav_v1.pickle
python main_results.py --config config/our.yaml --dataset data_processed/cifar10_swav_v1.pickle

# dino
python main_results.py --config config/our.yaml --dataset data_processed/matek_dino_v2.pickle
python main_results.py --config config/our.yaml --dataset data_processed/isic_dino_v2.pickle
python main_results.py --config config/our.yaml --dataset data_processed/jurkat_dino_v2.pickle
python main_results.py --config config/our.yaml --dataset data_processed/cifar10_dino_v2.pickle
