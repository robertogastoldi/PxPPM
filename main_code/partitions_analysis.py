import numpy as np
from scipy.stats import entropy

def compute_entropy(labels, base=None, label_format="list"):
    """
    Computes the entropy of a clustering result.

    Parameters:
        labels (list of lists or np.ndarray): The clustering information.
            - If `label_format` is "list", `labels` should be a list of lists, where each sublist 
              contains indices of points in a cluster.
            - If `label_format` is "array", `labels` should be a 1D array where each element represents 
              the cluster assignment of a point.
        base (float, optional): The logarithm base to use for entropy computation. Defaults to None, 
                                which means the natural logarithm is used.
        label_format (str, optional): The format of the `labels` input. Can be either "list" (list of lists) 
                                      or "array" (numpy array of cluster assignments). Defaults to "list".

    Returns:
        float: The entropy of the clustering result.
    """
    
    if label_format == "list":
        counts = []
        for cluster in labels:
            counts.append(len(cluster))
        return entropy(counts, base=base)
    if label_format == "array":
        _, counts = np.unique(labels, return_counts=True)
        return entropy(counts, base=base)  
