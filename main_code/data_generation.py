import numpy as np

def generate_gaussian_mixture(means, covariances, n_samples):
    """
    Generates synthetic data from a mixture of Gaussian distributions.

    Parameters:
        means (list of array-like): A list of mean vectors, where each vector represents the mean 
                                    of a Gaussian component.
        covariances (list of array-like): A list of covariance matrices, where each matrix defines 
                                          the spread of a Gaussian component.
        n_samples (list of int): A list specifying the number of samples to generate for each Gaussian component.

    Returns:
        data (np.ndarray): An array of generated data points, where each row represents a sample.
        labels (list of lists): A list of lists containing cluster membership indices, where each sublist 
                                corresponds to the indices of points belonging to a specific Gaussian component.
    """

    data = []
    labels = []

    start_idx = 0
    for i, (mean, cov, n) in enumerate(zip(means, covariances, n_samples)):
        points = np.random.multivariate_normal(mean, cov, n)
        data.append(points)
        cluster_indices = list(range(start_idx, start_idx + n))
        labels.append(cluster_indices)
        start_idx += n

    data = np.vstack(data)
    return data, labels
