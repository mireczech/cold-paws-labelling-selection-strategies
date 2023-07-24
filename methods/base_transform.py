
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from pdb import set_trace as pb
import pandas as pd
import importlib


def TSNE_transform(data, seed=0, iterations=1000, PCA=False, metric='euclidean'):
    if PCA:
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(data)
    else:
        pca_result = data

    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=iterations, random_state=seed, metric=metric, n_jobs=-1)
    tsne_results = tsne.fit_transform(pca_result)

    return tsne_results


matplotlib_present = importlib.util.find_spec("matplotlib")
if matplotlib_present is not None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    def plot_transform(data, indices, labels, save_file):

        df_subset = pd.DataFrame()
        df_subset['tsne-2d-one'] = data[:,0]
        df_subset['tsne-2d-two'] = data[:,1]
        df_subset['y'] = labels
        
        fig, ax = plt.subplots(figsize=(10,6))

        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            # hue='labelled',
            hue="y",
            palette=sns.color_palette("hls", np.unique(labels).shape[0]),
            data=df_subset,
            legend="full",
            alpha=0.3,
            ax=ax
        )

        if indices is not None:
            df_subset_labelled = df_subset.iloc[indices]
            sns.scatterplot(
                x="tsne-2d-one", y="tsne-2d-two",
                # hue='labelled',
                data=df_subset_labelled,
                legend="full",
                alpha=1.0,
                ax=ax,
                s=100, color=".2", marker="x"
            )

        plt.xlabel('')
        plt.ylabel('')
        plt.savefig(save_file, bbox_inches='tight')
