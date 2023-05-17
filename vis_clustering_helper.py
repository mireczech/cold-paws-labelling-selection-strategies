from methods.base_class import ExperimentImutable
from methods.base_class_greedy import ExperimentStatsGreedy
from pdb import set_trace as pb
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# https://builtin.com/data-science/tsne-python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

def plot_clustering(val_file, labelled_points='', labelled_points2='', label_order=None, seed=0, 
        iterations=1000, perplexity=40, n_components=2, metric='euclidean', 
        name_suffix='', tsne=True, encodings_metric='euclidean', num_points=50000,
        manual_labels='', labels_col='new_labels'):

    if labelled_points != '':
        lab_pts = pd.read_csv(labelled_points)
    if labelled_points2 != '':
        lab_pts2 = pd.read_csv(labelled_points2)

    ff = open('data_processed/'+val_file+'.pickle', 'rb')
    cifar10_data = pickle.load(ff)
    ff.close()

    if manual_labels != '':
        labels = pd.read_csv(manual_labels)
        if labels_col == 'new_labels':
            labels = labels['new_labels'].values
            labels = labels + 1
        else:
            labels = labels[labels_col].values
    elif hasattr(cifar10_data, 'labels'):
        labels = cifar10_data.labels
    else:
        labels = None
    print(np.unique(labels, return_counts=True))

    cifar10 = ExperimentStatsGreedy(cifar10_data.data, labels, trim=0.0)

    # pca = PCA(n_components=3)
    # pca_result = pca.fit_transform(df[feat_cols].values)

    np.random.seed(seed)
    rand = np.random.permutation(cifar10_data.data.shape[0])

    if (labelled_points != '') and (labelled_points2 != ''):
        indices = np.hstack((lab_pts['indices'].values, lab_pts2['indices'].values, rand[:num_points]))
        indices = np.unique(indices)
    elif labelled_points != '':
        indices = np.hstack((lab_pts['indices'].values, rand[:num_points]))
        indices = np.unique(indices)
    else:
        indices = rand[:num_points]

    if tsne:
        df_subset = cifar10_data.data[indices]
        tsne = TSNE(n_components=n_components, verbose=1, perplexity=perplexity, n_iter=iterations, metric=metric)
        tsne_results = tsne.fit_transform(df_subset)
    else:
        # tsne_results = pd.read_csv(encodings_file)
        if encodings_metric == 'euclidean':
            tsne_results = cifar10_data.data_transformed_euclidean[0]
        else:
            tsne_results = cifar10_data.data_transformed_cosine[0]
        # tsne_results = np.loadtxt(encodings_file, skiprows=1, delimiter=',')
        # tsne_results = tsne_results[:, 1:3]
        tsne_results = tsne_results[indices, :]

    labels = labels[indices]
    if labelled_points != '':
        indices_selected = np.array([x in lab_pts['indices'].values for x in indices])
    if labelled_points2 != '':
        indices_selected2 = np.array([x in lab_pts2['indices'].values for x in indices])

    df_subset = pd.DataFrame()

    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    df_subset['y'] = labels.astype(str)
    if label_order is not None:
        df_subset['y'] = np.array(label_order)[labels]
    if labelled_points != '':
        df_subset['labelled'] = indices_selected
        df_subset_labelled = df_subset[df_subset['labelled'] == True]
    if labelled_points2 != '':
        df_subset['labelled2'] = indices_selected2
        df_subset_labelled2 = df_subset[df_subset['labelled2'] == True]

    fig, ax = plt.subplots(figsize=(10,6))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        # hue='labelled',
        hue="y",
        palette=sns.color_palette("hls", np.unique(df_subset['y']).shape[0]),
        data=df_subset,
        legend="full",
        alpha=0.3,
        ax=ax
    )

    if labelled_points != '':
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            # hue='labelled',
            data=df_subset_labelled,
            legend="full",
            alpha=1.0,
            ax=ax,
            s=100, color=".2", marker="x"
        )

    if labelled_points2 != '':
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            # hue='labelled',
            data=df_subset_labelled2,
            legend="full",
            alpha=1.0,
            ax=ax,
            s=100, color=".2", marker="o", facecolors='none', edgecolor='black'
        )

    ax.legend(fancybox=True, framealpha=0.35)
    ax.set(xticklabels=[])  # remove the tick labels
    ax.set(yticklabels=[])  # remove the tick labels
    ax.tick_params(bottom=False, left=False)  # remove the ticks    plt.set(yticklabels=[])  # remove the tick labels

    plt.xlabel('')
    plt.ylabel('')
    plt.savefig('visualizations/clustering_'+val_file+'_'+name_suffix+'.png', bbox_inches='tight')

