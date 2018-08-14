import numpy as np
from sklearn.utils import linear_assignment_
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages


def acc_score(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
    accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def metrics(y_true, y_pred, verbose=True):
    assert len(y_true) == len(y_pred), 'illegal inputs'
    acc = acc_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    arc = adjusted_rand_score(y_true, y_pred)
    if verbose: print("acc: {:0.4f}, nmi: {:0.4f}, arc: {:0.4f}".format(acc, nmi, arc))
    return {'acc':acc, 'nmi':nmi, 'arc':arc}


def clustering(inputs, n_clusters, seed=42):
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4).fit(inputs)
    centroids = kmeans.cluster_centers_
    y_pred = kmeans.labels_

    return y_pred, centroids

def plot_latent_z_space(latent_z, labels, filepath, with_legend=False):
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(latent_z)
    n_samples = 10000
    fig, ax = plt.subplots()
    #for i in range(len(np.unique(labels))):
    #    idx_i = np.where(labels[:n_samples] == i)[0]
    #    print(idx_i.shape)
    #    plt.scatter(z_2d[idx_i, 0], z_2d[idx_i, 1], c=labels[idx_i], s=10, cmap='rainbow', alpha=0.5, label='%d' % i)
    #if with_legend: plt.legend()
    plt.scatter(z_2d[:n_samples, 0], z_2d[:n_samples, 1], c=labels[:n_samples], s=10, cmap='rainbow', alpha=0.5)
    #plt.savefig(filepath)
    #plt.close()

    pp = PdfPages('%s.pdf' % filepath)
    pp.savefig()
    pp.close()

    plt.close()




