import numpy as np
from tqdm import tqdm
import pickle
import random
import copy
from scipy.stats import multivariate_t
from scipy.spatial.distance import mahalanobis
from main_code.partitions_analysis import compute_entropy

class Neal_3:
    """
    Class for PPM based on Neal Algorithm 3
    """

    def __init__(self, alpha=0.1, lamb_0=1):
        
        # Initialize attributes

        # These use default values that can be changed
        self.alpha = alpha
        self.lamb_0 = lamb_0

        # These require data format to initialize
        self.Y = None
        self.D = None
        self.nu_0 = None  # Should be > D
        self.mu_0 = None  # Will default to mean of data
        self.inv_scale_mat_0 = None # Requires D

        # Attributes for computed when fitting data
        self.history = None
        self.similatity_matrix = None

        # Saving the point estimate computed via Binder loss minimization
        self.optimal_clustering = None
        self.optimal_loss = None
        
        self.metrics = {"entropy":[]}
        return 


    # Model hyper-parameter functions
    def compute_nu_0(self):
        """
        Computes and set the value for nu_0 depending on the number of dimensions D of Y.
        """

        if self.D is None:
            raise ValueError("No dimension D provided to compute nu_0")

        self.nu_0 = self.D + 3
    
    def compute_mu_0(self):
        """
        Computes and set the value for mu_0 as the mean of the data.
        """

        if self.Y is None:
            raise ValueError("No data Y provided to compute mu_0")
        
        self.mu_0 = np.mean(self.Y, axis=0)

    def compute_inv_scale_mat_0(self):
        """
        Computes the inverse scale matrix hyper parameter for the NIW.
        Defaults as the identity.
        """

        self.inv_scale_mat_0 = np.eye(self.D)


    # Integrals based on data distribution assumptions
    def integral_func_1(self, cluster, i):
        """
        Computes the first integral using the Student-t distribution based on Murphy (2007) parameters.

        Parameters:
            cluster (list of int): A list of observation indices representing the current cluster.
            i (int): The index of the observation for which the integral is computed.

        Returns:
            float: The computed integral value based on the Student-t probability density function.
        """

        n = len(cluster)    # Number of element currently in cluster (used n to be consistent with Murphy (2007) notation)
        
        cluster_Y = self.Y[np.isin(np.arange(self.n_obs), cluster)]
        cluster_mean = np.mean(cluster_Y, axis=0)

        # Based on Murphy (2007)
        mu_n = (self.lamb_0 * self.mu_0 + n * cluster_mean) / (self.lamb_0 + n)
        lamb_n = self.lamb_0 + n
        nu_n = self.nu_0 + n
        
        # Compute scatter matrix
        S = np.zeros((self.D,self.D))
        for j in range(n):
            temp = self.Y[j] - cluster_mean
            S += np.outer(temp, temp)
        temp = cluster_mean - self.mu_0
        inv_scale_mat_n = self.inv_scale_mat_0 + S + ((self.lamb_0 * n) / (self.lamb_0 + n)) * np.outer(temp, temp)

        # Computes integral using pdf of student t
        student_df = nu_n - self.D + 1
        integral = multivariate_t.pdf(self.Y[i],
                                    mu_n,
                                    inv_scale_mat_n * ((lamb_n+1) / (lamb_n * student_df)),
                                    student_df)
        return integral
    
    def integral_func_2(self, i):
        """
        Computes the second integral using the Student-t distribution based on Murphy (2007) parameters.

        Parameters:
            i (int): The index of the observation for which the integral is computed.

        Returns:
            float: The computed integral value based on the Student-t probability density function.
        """

        student_df = self.nu_0 - self.D + 1

        # Computes integral using pdf of student t
        integral = multivariate_t.pdf(self.Y[i],
                                    self.mu_0,
                                    self.inv_scale_mat_0 * ((self.lamb_0 + 1) / (self.lamb_0 * student_df)),
                                    student_df)
        return integral


    # Functions for Neal Algorithm 3
    def cluster_probabilities(self, i, clusters):
        """
        Computes the weights for an observation joining existing clusters or creating a new one.

        Parameters:
            i (int): The index of the observation to evaluate.
            clusters (list of lists): The current partitioning of observations, where each sublist 
                                    contains indices of points belonging to a cluster.

        Returns:
            np.ndarray: An array of weights representing the likelihood of observation `i` joining 
                        each existing cluster or forming a new one. The last element corresponds to 
                        the weight of creating a new cluster.
        """

        n_clusters = len(clusters)
        probabilities = np.zeros(n_clusters+1)

        # Probabilities of joining existing cluster
        for c in range(n_clusters):
            probabilities[c] = self.integral_func_1(clusters[c], i)
            probabilities[c] *= (len(clusters[c]) / (self.n_obs - 1 + self.alpha))

        # Probability of creating new cluster
        probabilities[-1] = self.integral_func_2(i)
        probabilities[-1] *= self.alpha / (self.n_obs - 1 + self.alpha)

        return probabilities

    def fit(self, Y, n_steps, metrics=["entropy"]):
        """
        Performs Markov Chain Monte Carlo (MCMC) clustering using Algorithm 3 from Neal (2000).

        Parameters:
            Y (np.ndarray): A 2D array of observations, where each row represents an observation 
                            and each column represents a feature. Shape is (n_observations, D).
            n_steps (int): The number of MCMC steps to perform. One step consists of randomly 
                        moving each observation once.
            metrics (list of str, optional): A list of metric names to compute during runtime. 
                                            Currently, only "entropy" is implemented. Defaults to ["entropy"].

        Returns:
            list of lists: A history of partitions at each step of the Markov chain, where each 
                        partition is represented as a list of clusters (each cluster is a list 
                        of observation indices).
        """

        # Set basic attributes
        self.Y = Y
        self.n_obs = len(Y)
        self.D = Y.shape[1]
        self.compute_mu_0()
        self.compute_inv_scale_mat_0()
        self.compute_nu_0()

        # Initialize clusters
        clusters = [[i] for i in range(self.n_obs)]

        self.history = [copy.deepcopy(clusters)]

        # Update_metrics
        self.update_metrics(metrics, clusters)

        # Initialize progress bar
        progress_bar = tqdm(total=n_steps, desc="MCMC Progress", unit="step")

        for step in range(n_steps):  # Markov chain
            for i in range(self.n_obs):  # 1 step of the Markov chain
                # 1. Find in which cluster the observation is
                c = 0
                for index in range(len(clusters)):
                    if i in clusters[index]:
                        c = index
                        break
                # 2. Remove observation i from clusters:
                if len(clusters[c]) == 1:  # Case 1: i is the only element of the cluster -> remove cluster
                    del clusters[c]
                else:  # Case 2: cluster has more than 1 element -> remove i from the cluster
                    clusters[c].remove(i)

                # 3. Compute probabilities of adding i to each cluster
                weights = self.cluster_probabilities(i, clusters)
                transitions = list(range(len(weights)))
                transition = random.choices(transitions, weights=weights)[0]

                # 4. Apply transition 
                if transition == len(clusters):  # add new cluster
                    clusters.append([i])
                else:
                    clusters[transition].append(i)
            
            # All elements have moved once -> one step of the Markov chain
            self.history.append(copy.deepcopy(clusters))
            
            # Update_metrics
            self.update_metrics(metrics, clusters)

            # Update progress bar
            progress_bar.update(1)

        # Close progress bar
        progress_bar.close()

        return self.history    

    # Functions for metrics
    def update_metrics(self, metrics, clusters):
        """
        Updates the specified metrics during MCMC clustering.

        Parameters:
            metrics (list of str): A list of metric names to compute. Currently supports "entropy".
            clusters (list of lists): The current partitioning of observations, where each sublist 
                                    contains indices of points belonging to a cluster.

        Returns:
            None: Updates the stored metric values in `self.metrics` during the MCMC process.
        """

        if "entropy" in metrics:
            entropy = compute_entropy(clusters, label_format="list")
            self.metrics["entropy"].append(entropy)
        return

    # Post Processing functions :
    def compute_similarity_matrix(self, burn_in=0):
        """
        Computes the similarity matrix based on the MCMC clustering history.

        Parameters:
            burn_in (int, optional): The number of initial iterations to discard before computing 
                                    the similarity matrix. Defaults to 0.

        Returns:
            np.ndarray: A (n_obs, n_obs) similarity matrix, where each entry (i, j) represents 
                        the proportion of MCMC samples in which observations i and j were clustered together.
        """

        if self.history is None:
            raise RuntimeError("No MCMC history to compute the similarity matrix")
        A = np.zeros((self.n_obs, self.n_obs), dtype=float)
        n_samples = len(self.history)

        # Initialize progress bar
        progress_bar = tqdm(total=len(self.history[burn_in:]), desc="Similarity Matrix Progress", unit="step")

        for clusters in self.history[burn_in:]:
            for cluster in clusters:
                for k, i in enumerate(cluster):
                    for j in cluster[k:]:
                        if i != j:  # Avoid double increment for diagonal
                            A[i, j] += 1
                            A[j, i] += 1

            # Update progress bar
            progress_bar.update(1)

        # Close progress bar
        progress_bar.close()

        # Normalize by the number of samples
        A /= n_samples

        # Ensure the diagonal is 1 (observations are always in the same cluster with themselves)
        np.fill_diagonal(A, 1.0)

        # Both save the matrix and return it
        self.similatity_matrix = A
        return A
    
    def binder_loss(self, clustering, alpha=1.0, beta=1.0):
        """
        Computes the Binder loss for a given clustering.

        Parameters:
            clustering (array-like or list of lists): The clustering representation.
                - If array-like, clustering[i] represents the cluster label of observation i.
                - If list of lists, each sublist contains indices of data points belonging to the same cluster.
            alpha (float, optional): Weight for within-cluster disagreements. Defaults to 1.0.
            beta (float, optional): Weight for between-cluster disagreements. Defaults to 1.0.

        Returns:
            float: The computed Binder loss value, representing the disagreement between the clustering 
                and the posterior similarity matrix.
        """

        # If clustering is in list-of-lists format, convert to label format
        if isinstance(clustering, list):
            N = self.similatity_matrix.shape[0]
            labels = np.zeros(N, dtype=int)
            for cluster_id, indices in enumerate(clustering):
                for index in indices:
                    labels[index] = cluster_id
        else:
            labels = np.array(clustering)
        
        # Compute the Binder loss
        loss = 0.0
        N = self.similatity_matrix.shape[0]
        for i in range(N):
            for j in range(i + 1, N):
                same_cluster = labels[i] == labels[j]
                loss += alpha * same_cluster * (1 - self.similatity_matrix[i, j]) + beta * (not same_cluster) * self.similatity_matrix[i, j]
        return loss
    
    def find_optimal_clustering(self, alpha=1.0, beta=1.0):
        """
        Finds the clustering that minimizes the Binder loss.

        Parameters:
            alpha (float, optional): Weight for within-cluster disagreements. Defaults to 1.0.
            beta (float, optional): Weight for between-cluster disagreements. Defaults to 1.0.

        Returns:
            optimal_clustering (list of lists): The clustering configuration that minimizes the Binder loss.
            optimal_loss (float): The Binder loss value corresponding to the optimal clustering.
        """

        if self.similatity_matrix is None:
            raise ValueError("Similarity matrix not yet computed")
        
        # Set a very big number to initialize the loss
        self.optimal_loss = 1e9

        # Initialize progress bar
        progress_bar = tqdm(total=len(self.history), desc="Point Estimate Progress", unit="step")

        for clustering in self.history:
            loss = self.binder_loss(clustering, alpha, beta)
            if loss < self.optimal_loss:
                self.optimal_loss = loss
                self.optimal_clustering = clustering

            # Update progress bar
            progress_bar.update(1)

        # Close progress bar
        progress_bar.close()
        
        return self.optimal_clustering, self.optimal_loss
    
    def save(self, file_path):
        """
        Saves the current object to a file.

        Parameters:
            file_path (str): The path to the file where the object will be saved.

        Returns:
            None: The object is saved to the specified file.
        """

        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file_path):
        """
        Loads an object from a file.

        Parameters:
            file_path (str): The path to the file from which the object will be loaded.

        Returns:
            loaded_object: The object that was loaded from the file.
        """

        with open(file_path, 'rb') as file:
            loaded_object = pickle.load(file)
        return loaded_object


class PPMx(Neal_3):
    """
    Extends Neal_3 class by including covariates
    """

    def __init__(self, alpha=0.1, lamb_0=1):
        super().__init__(alpha=alpha, lamb_0=lamb_0)

        # Attributes specific to algorithm with covariates
        self.lambda_penalty = None
        self.X = None

    def compute_mahalanobis_penalty(self, cluster, i):
        """
        Computes the Mahalanobis distance penalty for adding an observation to a cluster.

        Parameters:
            cluster (list of int): Indices of the observations in the current cluster.
            i (int): Index of the new observation being evaluated.

        Returns:
            penalty (float): The Mahalanobis distance between the new observation and the cluster.
        """

        # Combine current cluster observations and the new observation
        cluster_data = np.array([self.X[idx] for idx in cluster] + [self.X[i]])
        cluster_mean = np.mean(cluster_data, axis=0)
        cov_matrix = np.cov(cluster_data.T)

        penalty = mahalanobis(self.X[i], cluster_mean, cov_matrix)
        return penalty

    def cluster_probabilities(self, i, clusters):
        """
        Computes the weights for an observation joining existing clusters or creating a new one, 
            incorporating a Mahalanobis distance penalty.

        Parameters:
            i (int): The index of the observation to evaluate.
            clusters (list of lists): The current partitioning of observations, where each sublist 
                                    contains indices of points belonging to a cluster.

        Returns:
            np.ndarray: An array of weights representing the likelihood of observation `i` joining 
                        each existing cluster or forming a new one, adjusted with the Mahalanobis penalty. 
                        The last element corresponds to the weight of creating a new cluster.
        """

        probabilities = super().cluster_probabilities(i, clusters)

        n_clusters = len(clusters)
        for c in range(n_clusters):
            penalty = self.compute_mahalanobis_penalty(clusters[c], i)
            probabilities[c] *= np.exp(-self.lambda_penalty * penalty)
        
        return probabilities
    
    def fit(self, Y, X, n_steps, lambda_penalty=0.1, metrics=["entropy"]):
        """
        Performs Markov Chain Monte Carlo (MCMC) clustering using Algorithm 3 from Neal (2000).

        Parameters:
            Y (np.ndarray): A 2D array of observations, where each row represents an observation 
                            and each column represents a feature. Shape is (n_observations, D).
            X (numpy.ndarray): Covariate matrix used for computing the Mahalanobis penalty.
            n_steps (int): The number of MCMC steps to perform. One step consists of randomly 
                        moving each observation once.
            lambda_penalty (float, optional): Weight for the Mahalanobis distance penalty. Defaults to 0.1.
                                              If set to 0, this algorithm is equivalent to Neal_3.
            metrics (list of str, optional): A list of metric names to compute during runtime. 
                                            Currently, only "entropy" is implemented. Defaults to ["entropy"].

        Returns:
            list of lists: A history of partitions at each step of the Markov chain, where each 
                        partition is represented as a list of clusters (each cluster is a list 
                        of observation indices).
        """

        self.X = X
        self.lambda_penalty = lambda_penalty
        return super().fit(Y, n_steps, metrics=metrics)



class Second_Layer(Neal_3):
    """
    Extends Neal_3 class by including covariates
    """

    def __init__(self, alpha=0.1, lamb_0=1):
        super().__init__(alpha=alpha, lamb_0=lamb_0)

        # Attributes specific to algorithm with covariates
        self.lambda_penalty = None
        self.X = None
        self.initial_partition = None
        self.n_partitions = None

    def compute_mahalanobis_penalty(self, cluster, clust):
        """
        Computes the Mahalanobis distance penalty for adding an observation to a cluster.

        Parameters:
            cluster (list of int): Indices of the observations in the current cluster.
            clust (list of int): Indeces of the new observations being evaluated.

        Returns:
            penalty (float): The Mahalanobis distance between the new observation and the cluster.
        """

        # Combine current cluster observations and the new observation
        cluster_data = np.array([self.X[idx] for idx in cluster] + [self.X[i] for i in clust])
        cluster_mean = np.mean(cluster_data, axis=0)
        cov_matrix = np.cov(cluster_data.T)

        actual_X = np.array([self.X[i] for i in clust])
        compress_X = np.mean(actual_X, axis=0)

        penalty = mahalanobis(compress_X, cluster_mean, cov_matrix)
        return penalty

def compute_kernels(self, cluster):
    """
        Computes the second integral using the Student-t distribution based on Murphy (2007) parameters.

        Parameters:
            cluster (list of int): A list of observations in cluster clust for which the kernel is computed.

        Returns:
            float: The computed kernel value based on the observations in the cluster.
    """
    n = len(cluster)    # Number of element currently in cluster (used n to be consistent with Murphy (2007) notation)
        
        cluster_Y = self.Y[np.isin(np.arange(self.n_obs), cluster)]
        cluster_mean = np.mean(cluster_Y, axis=0)

        # Based on Murphy (2007)
        mu_n = (self.lamb_0 * self.mu_0 + n * cluster_mean) / (self.lamb_0 + n)
        lamb_n = self.lamb_0 + n
        nu_n = self.nu_0 + n
        
        # Compute scatter matrix
        S = np.zeros((self.D,self.D))
        for j in range(n):
            temp = self.Y[j] - cluster_mean
            S += np.outer(temp, temp)
        temp = cluster_mean - self.mu_0
        inv_scale_mat_n = self.inv_scale_mat_0 + S + ((self.lamb_0 * n) / (self.lamb_0 + n)) * np.outer(temp, temp)

        # Computes integral using pdf of multivariate gaussian distribution
        kernel = 1
        for i in cluster:
            kernel *= multivariate_normal.pdf(self.Y[i], mean=mu_n, cov=inv_scale_mat_n)
        return kernel

def integral_func_1(self, cluster, clust):
        """
        Computes the first integral using the Student-t distribution based on Murphy (2007) parameters.

        Parameters:
            cluster (list of int): A list of observation indices representing the current cluster.
            clust (list of int): A list of observations in cluster clust.

        Returns:
            float: The computed integral value based on the Student-t probability density function.
        """

        n = len(cluster)    # Number of element currently in cluster (used n to be consistent with Murphy (2007) notation)
        
        cluster_Y = self.Y[np.isin(np.arange(self.n_obs), cluster)]
        cluster_mean = np.mean(cluster_Y, axis=0)

        # Based on Murphy (2007)
        mu_n = (self.lamb_0 * self.mu_0 + n * cluster_mean) / (self.lamb_0 + n)
        lamb_n = self.lamb_0 + n
        nu_n = self.nu_0 + n
        
        # Compute scatter matrix
        S = np.zeros((self.D,self.D))
        for j in range(n):
            temp = self.Y[j] - cluster_mean
            S += np.outer(temp, temp)
        temp = cluster_mean - self.mu_0
        inv_scale_mat_n = self.inv_scale_mat_0 + S + ((self.lamb_0 * n) / (self.lamb_0 + n)) * np.outer(temp, temp)

        # Computes integral using pdf of student t
        student_df = nu_n - self.D + 1

        # Summary statistics to pass to the integral
        actual_Y = np.array([self.Y[i] for i in clust])
        compress_Y = np.mean(actual_Y, axis=0)
    
        integral = multivariate_t.pdf(compress_Y,
                                    mu_n,
                                    inv_scale_mat_n * ((lamb_n+1) / (lamb_n * student_df)),
                                    student_df)
        return integral

def integral_func_2(self, clust):
        """
        Computes the second integral using the Student-t distribution based on Murphy (2007) parameters.

        Parameters:
            clust (list of int): A list of observations in cluster clust for which the integral is computed.

        Returns:
            float: The computed integral value based on the Student-t probability density function.
        """

        student_df = self.nu_0 - self.D + 1

        # Computes integral using pdf of student t
    
        # Summary statistics to pass to the integral
        actual_Y = np.array([self.Y[i] for i in clust])
        compress_Y = np.mean(actual_Y, axis=0)
    
        integral = multivariate_t.pdf(compress_Y,
                                    self.mu_0,
                                    self.inv_scale_mat_0 * ((self.lamb_0 + 1) / (self.lamb_0 * student_df)),
                                    student_df)
        return integral

def cluster_probabilities(self, clust, clusters):
        """
        Computes the weights for an observation joining existing clusters or creating a new one.

        Parameters:
            clust (list): The list of the observations belonging to clust.
            clusters (list of lists): The current partitioning of observations, where each sublist 
                                    contains indices of points belonging to a cluster.

        Returns:
            np.ndarray: An array of weights representing the likelihood of observation `i` joining 
                        each existing cluster or forming a new one. The last element corresponds to 
                        the weight of creating a new cluster.
        """

        n_clusters = len(clusters)
        probabilities = np.zeros(n_clusters+1)

        # Probabilities of joining existing cluster
        for c in range(n_clusters):
            probabilities[c] = self.integral_func_1(clusters[c], clust)
            probabilities[c] *= self.compute_kernels(clusters[c])
            probabilities[c] *= (len(clusters[c]) / (self.n_obs - 1 + self.alpha))

        # Probability of creating new cluster
        probabilities[-1] = self.integral_func_2(clust)
        probabilities[-1] = self.compute_kernels(clust)
        probabilities[-1] *= self.alpha / (self.n_obs - 1 + self.alpha)

        n_clusters = len(clusters)
        for c in range(n_clusters):
            penalty = self.compute_mahalanobis_penalty(clusters[c], clust)
            probabilities[c] *= np.exp(-self.lambda_penalty * penalty)

        return probabilities

def fit(self, Y, X, initial_partition, n_steps, lambda_penalty=0.1, metrics=["entropy"]):
    """
    Parameters:
        Y (np.ndarray): A 2D array of observations, where each row represents an observation 
                        and each column represents a feature. Shape is (n_observations, D).
        X (numpy.ndarray): Covariate matrix used for computing the Mahalanobis penalty.
        initial_partition (list of list): Optimal partition of the 1st layer.
        n_steps (int): The number of MCMC steps to perform. One step consists of randomly 
                    moving each observation once.
        lambda_penalty (float, optional): Weight for the Mahalanobis distance penalty. Defaults to 0.1.
                                          If set to 0, this algorithm is equivalent to Neal_3.
        metrics (list of str, optional): A list of metric names to compute during runtime. 
                                        Currently, only "entropy" is implemented. Defaults to ["entropy"].

    Returns:
        list of lists: A history of partitions at each step of the Markov chain, where each 
                    partition is represented as a list of clusters (each cluster is a list 
                    of observation indices).
    """
    # Set basic attributes
    self.X = X
    self.lambda_penalty = lambda_penalty
    self.initial_partition = initial_partition
    self.n_partitions = len(initial_partition)
    self.Y = Y
    self.n_obs = len(Y)
    self.D = Y.shape[1]
    self.compute_mu_0()
    self.compute_inv_scale_mat_0()
    self.compute_nu_0()

    # Initialize clusters
    clusters = copy.deepcopy(initial_partition)  # Evita modifiche accidentali all'originale

    self.history = [copy.deepcopy(clusters)]

    # Update metrics
    self.update_metrics(metrics, clusters)

    # Initialize progress bar
    progress_bar = tqdm(total=n_steps, desc="MCMC Progress", unit="step")

    for step in range(n_steps):  # Markov chain

        for clust in initial_partition:
            # 1. Trova il cluster che contiene `clust`
            c = next((index for index, cluster in enumerate(clusters) if set(clust) == set(cluster)), None)

            # 2. Rimuove `clust` da `clusters`
            if set(clusters[c]) == set(clust):  # Se `clust` è l'unico elemento nel cluster
                del clusters[c]  # Rimuove l'intero cluster
            else:  # Se ci sono altri elementi nel cluster, rimuove solo `clust`
                clusters[c] = [x for x in clusters[c] if x not in clust]

            # 3. Calcola le probabilità di assegnazione del cluster
            weights = self.cluster_probabilities(clust, clusters)
            transitions = list(range(len(weights)))
            transition = random.choices(transitions, weights=weights)[0]

            # 4. Applica la transizione
            if transition == len(clusters):  # Se viene creato un nuovo cluster
                clusters.append(clust)
            else:
                clusters[transition].extend(clust)  # Aggiunge `clust` a un cluster esistente
        
        # Fine del passo MCMC
        self.history.append(copy.deepcopy(clusters))
        
        # Update metrics
        self.update_metrics(metrics, clusters)

        # Update progress bar
        progress_bar.update(1)

    # Chiudi progress bar
    progress_bar.close()

    return self.history

    
