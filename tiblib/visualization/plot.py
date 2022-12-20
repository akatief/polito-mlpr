import numpy as np
from warnings import warn
from matplotlib import pyplot as plt


def pair_plot(X,y,columns=None):
    nrows = X.shape[0]
    ncols = X.shape[1]
    labels = np.unique(y)
    if ncols > nrows:
        warn(f'Samples in X should be rows. Are you sure the dataset is not transposed? Size: {X.shape}')
    if columns is not None:
        assert len(columns) == ncols, 'Size of X does not match number of column names provided'
    fig, axs = plt.subplots(4,4)
    fig.set_size_inches(18.5, 10.5)
    for feat1 in range(ncols):
        for feat2 in range(ncols):
            for l in labels:
                if feat1 == feat2:
                    axs[feat1, feat2].hist(X[y == l][:,feat1], alpha=0.3)
                    if columns is not None:
                        axs[feat1, feat2].set_xlabel(columns[feat1])
                else:
                    axs[feat1, feat2].scatter(X[y == l][:,feat1],X[y == l][:,feat2],marker='.')
                    if columns is not None:
                        axs[feat1, feat2].set_xlabel(columns[feat1])
                        axs[feat1, feat2].set_ylabel(columns[feat2])
    plt.tight_layout()
    plt.show()
