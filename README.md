# Cold PAWS: Unsupervised class discovery and addressing the cold-start problem for semi-supervised learning
## Repository to reproduce label selection results

This is python code for the label selection strategies from the [paper](https://arxiv.org/abs/2305.10071). For the model fitting code see [this repo](https://github.com/emannix/cold-paws-simclr-and-paws-semi-supervised-learning).

---

First, download the sample data from [here](https://drive.google.com/file/d/1NzFKnz438yZ9sLrhUbtuJBzYxG8iLBTD/view?usp=share_link). Then, to initialise t-SNE clusters and the data files, run

```bash
python main_setup.py
```

The next step is to select indices using the methods defined the config files. In this instance, 'finetune' is the mini-max approach and 'repulsive' is the maxi-min approach. To run these methods on CIFAR-10, for a budget of 40 labels, use

```bash
python main_results.py --config config/test.yaml --dataset data_processed/cifar10.pickle
```

Here are some further examples to generate the benchmarking runs, or the results for the imagenette dataset

```bash
python main_results.py --config config/benchmark.yaml --dataset data_processed/sw24708.pickle
python main_results.py --config config/base-extra-class-disc.yaml --dataset data_processed/imagenette.pickle
```

To take the output files from a particular run and put them all together in a CSV file, run

```bash
python unsupervised_class_detection.py --config config/benchmark.yaml --processed_data 'data_processed/sw24708.pickle'
```

To make t-SNE plots from the encoding files in data, run

```bash
python vis_clustering.py
```

The output folder contains some example outputs for reference which need to be downloaded seperately from the link above.
