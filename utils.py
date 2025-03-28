from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import silhouette_score
from sklearn.utils import check_X_y
import numpy as np



def dbscan_scorer_before(estimator, X):
    #use labels
    labels = estimator.named_steps['dbscan'].labels_
    
    # Filter noise (-1)
    valid_labels = labels[labels != -1]
    
    #must have at least one cluster
    if len(set(valid_labels)) > 1:
        return silhouette_score(X, labels[labels != -1], random_state=42)
    else:
        # Return -1 to show no cluster is found
        return -1
    

def dbscan_scorer(estimator, X, labels):
    # Count of non-noise points
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    # If all points are noise or only one cluster, return a bad score
    if n_clusters <= 1:
        return -1.0
    
    # Calculate silhouette score only on non-noise points if there are any
    if n_noise < len(labels):
        # Create a mask for non-noise points
        mask = labels != -1
        if sum(mask) > 1:  # Need at least 2 points for silhouette
            try:
                score = silhouette_score(X[mask], labels[mask])
                # We want fewer noise points for same silhouette
                noise_penalty = n_noise / len(labels)
                return score * (1 - noise_penalty)
            except:
                return -1.0
    
    return -1.0