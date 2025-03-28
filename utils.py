from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import silhouette_score
from sklearn.utils import check_X_y
import numpy as np



def dbscan_scorer(estimator, X):
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