from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

def evaluate_clustering(X, labels):
    results = {}

    results["silhouette"] = silhouette_score(X, labels)
    results["calinski_harabasz"] = calinski_harabasz_score(X, labels)
    results["davies_bouldin"] = davies_bouldin_score(X, labels)

    return results
