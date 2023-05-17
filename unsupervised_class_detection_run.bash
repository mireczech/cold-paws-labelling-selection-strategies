
python unsupervised_class_detection.py --config config/base-extra-class-disc.yaml --processed_data 'data_processed/cifar10-resnet18sk0-1024-normed-ph.pickle'
python unsupervised_class_detection.py --config config/base-extra-class-disc.yaml --processed_data 'data_processed/cifar10-resnet18sk0-1024-normed-bph.pickle'

python unsupervised_class_detection.py --config config/base-extra-class-disc.yaml --processed_data 'data_processed/imagenette-resnet18sk0-bignormed-rw-ph.pickle'
python unsupervised_class_detection.py --config config/base-extra-class-disc.yaml --processed_data 'data_processed/imagenette-resnet18sk0-bignormed-rw-bph.pickle'

python unsupervised_class_detection.py --config config/base-extra-class-disc-9.yaml --processed_data 'data_processed/deepweeds-resnet18sk0-normed-dedup-ph.pickle'
python unsupervised_class_detection.py --config config/base-extra-class-disc-9.yaml --processed_data 'data_processed/deepweeds-resnet18sk0-normed-dedup-bph.pickle'

python unsupervised_class_detection.py --config config/base-extra-class-disc.yaml --processed_data 'data_processed/eurosat-resnet18sk0-normed-dedup-ph.pickle'
python unsupervised_class_detection.py --config config/base-extra-class-disc.yaml --processed_data 'data_processed/eurosat-resnet18sk0-normed-dedup-bph.pickle'


python unsupervised_class_detection.py --config config/benchmark.yaml --processed_data 'data_processed/bm33708.pickle'
python unsupervised_class_detection.py --config config/benchmark.yaml --processed_data 'data_processed/ch71009.pickle'
python unsupervised_class_detection.py --config config/benchmark.yaml --processed_data 'data_processed/sw24708.pickle'
