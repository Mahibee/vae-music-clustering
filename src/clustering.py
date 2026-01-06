from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

def apply_kmeans(X, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels

def visualize_tsne(X, labels, save_path):
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)

    X_2d = tsne.fit_transform(X)

    plt.figure()
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels)
    plt.title("t-SNE Visualization of Clusters")
    plt.savefig(save_path)
    plt.close()
