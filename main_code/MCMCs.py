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

        # Attributes specific to super-cluster structure
        self.shistory = None

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

class PxPPM(Neal_3):

    def __init__(self, alpha=1.0, alpha_super=1.0, lamb_0=1.0, kappa_0_super=1.0, lambda_penalty=1.0):
        # initialization of the base class
        super().__init__(alpha=alpha, lamb_0=lamb_0)
        
        # Livello 2 parameters
        self.alpha_super = alpha_super  # Propensity to create new super-cluster
        self.kappa_0_super = kappa_0_super # The “strength” of the prior on the covariates level
        self.lambda_penalty = lambda_penalty
        
        self.mu_0_super = None
        self.nu_0_super = None
        self.inv_scale_super = None
        self.D_super = None

    # Modify integral_func_1 to handle n=0
    def integral_func_1(self, cluster, i):
        n = len(cluster)
        
        if n == 0:
            # If the cluster is empty, the posterior is equal to the prior.
            mu_n = self.mu_0
            lamb_n = self.lamb_0
            nu_n = self.nu_0
            inv_scale_mat_n = self.inv_scale_mat_0
        else:
            cluster_Y = self.Y[np.isin(np.arange(self.n_obs), cluster)]
            cluster_mean = np.mean(cluster_Y, axis=0)
            mu_n = (self.lamb_0 * self.mu_0 + n * cluster_mean) / (self.lamb_0 + n)
            lamb_n = self.lamb_0 + n
            nu_n = self.nu_0 + n
            diff = cluster_Y - cluster_mean
            S = np.dot(diff.T, diff)
            temp = cluster_mean - self.mu_0
            inv_scale_mat_n = self.inv_scale_mat_0 + S + ((self.lamb_0 * n) / (self.lamb_0 + n)) * np.outer(temp, temp)

        student_df = nu_n - self.D + 1
        scale_matrix = inv_scale_mat_n * ((lamb_n + 1) / (lamb_n * student_df))
        
        # Enforce symmetry to avoid numerical precision errors
        scale_matrix = (scale_matrix + scale_matrix.T) / 2
        
        return multivariate_t.pdf(self.Y[i], mu_n, scale_matrix, student_df)
    
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
    
    def _sync_remove_subcluster(self, sub_idx_dead, super_clusters):
        """ 
        When a sub-cluster exits Level 1, it must disappear from Level 2,
        and all indexes > sub_idx_dead must be decremented by -1.
        """
        # 1.Remove from super-clusters
        for k in range(len(super_clusters) - 1, -1, -1):
            if sub_idx_dead in super_clusters[k]:
                super_clusters[k].remove(sub_idx_dead)
                if len(super_clusters[k]) == 0:
                    del super_clusters[k] # Rimuovi super-cluster vuoto
        
        # 2. Index shifts to maintain consistency
        for k in range(len(super_clusters)):
            for i in range(len(super_clusters[k])):
                if super_clusters[k][i] > sub_idx_dead:
                    super_clusters[k][i] -= 1

    def _calc_mahalanobis_similarity(self, X_j, X_k_all):
        """
        Calculate similarity based on Mahalanobis distance.
        """
        # Mean of covariates in sub-cluster j
        mu_j = np.mean(X_j, axis=0)
        
        # Statistics of the super-cluster k
        mu_k = np.mean(X_k_all, axis=0)
        
        # Covariance of supercluster k 
        if len(X_k_all) > 1:
            cov_k = np.cov(X_k_all.T) + np.eye(self.D_super) * 1e-4
        else:
            # If the supercluster has only one point, we use the global variance.
            cov_k = self.inv_scale_super + np.eye(self.D_super) * 1e-4
            
        inv_cov_k = np.linalg.inv(cov_k)
        
        # Mahalanobis distance squared
        delta = mu_j - mu_k
        dist_sq = np.dot(np.dot(delta, inv_cov_k), delta.T)
        
        # We transform the distance into a “log-likelihood.”
        return -0.5 * dist_sq

    def _get_super_probabilities(self, j_idx, super_clusters, sub_clusters):
        log_weights = []
        punti_j = sub_clusters[j_idx]
        X_j = self.X[punti_j]

        for s_indices in super_clusters:
            # 1. Prior
            log_prior = np.log(len(s_indices))

            # 2. Spatial Likelihood
            # We want to consider only the effect of the covariates, set log_lik_Y = 0
            log_lik_Y = 0 

            # 3. Covariates similarity (Mahalanobis)
            X_k_all = []
            for s_idx in s_indices:
                X_k_all.extend(self.X[sub_clusters[s_idx]])
            X_k_all = np.array(X_k_all)
            
            log_sim_X = self._calc_mahalanobis_similarity(X_j, X_k_all)

            log_weights.append(log_prior + log_lik_Y + log_sim_X)

        # New super-cluster case
        # We use a “default” distance or one based on prior
        log_prior_new = np.log(self.alpha_super)
        log_sim_X_new = -self.D_super  # Fixed penalty for creating a new group
        
        log_weights.append(log_prior_new + log_sim_X_new)

        # Normalization
        max_log = np.max(log_weights)
        probs = np.exp(log_weights - max_log)
        return probs / np.sum(probs)

    def fit(self, Y, X, n_steps, metrics=["entropy"]):
        """
        Performs Markov Chain Monte Carlo (MCMC) clustering using PxPPM.

        Parameters:
            Y (np.ndarray): A 2D array of observations, where each row represents an observation 
                            and each column represents a feature. Shape is (n_observations, D).
            X (np.ndarray): A 2D array of observations, where each row represents an observation
                            and each column represents a covatiate. Shape id (n_observations, D)
            n_steps (int): The number of MCMC steps to perform. One step consists of randomly 
                        moving each observation once.
            metrics (list of str, optional): A list of metric names to compute during runtime. 
                            Currently, only "entropy" is implemented. Defaults to ["entropy"].

        Returns:
            list of lists: A history of sub-partitions at each step of the Markov chain, where each 
                        partition is represented as a list of clusters (each cluster is a list 
                        of observation indices).
            list of lists: A history of super-partitions at each step of the Markov chain, where each 
                        partition is represented as a list of super-clusters (each super-cluster is 
                        a list of sub-clusters indices).
        """
        # Setup Level 1 (Y coordinates)
        self.Y = Y
        self.n_obs = len(Y)
        self.D = Y.shape[1]
        self.compute_mu_0()
        self.compute_inv_scale_mat_0()
        self.compute_nu_0()
        
        # Setup Level 2 (X Covariates)
        self.X = X
        self.D_super = X.shape[1]
        self.mu_0_super = np.mean(X, axis=0)
        self.inv_scale_super = np.diag(np.var(X, axis=0)) * 0.1
        self.nu_0_super = self.D_super + 2

        #Initialization
        # Each point is assigned to its own sub-cluster.
        sub_clusters = [[i] for i in range(self.n_obs)]
        # Each sub-cluster forms a super-cluster.
        super_clusters = [[j] for j in range(len(sub_clusters))]
        self.update_metrics(metrics, sub_clusters)
        
        self.history = [copy.deepcopy(sub_clusters)]
        self.shistory = [copy.deepcopy(super_clusters)]

        pbar = tqdm(total=n_steps, desc="PxPPM")
        
        for step in range(n_steps):
            
            # Level 1: PPMx on observations
            for i in range(self.n_obs):
                c_idx = next(idx for idx, cl in enumerate(sub_clusters) if i in cl)
                
                if len(sub_clusters[c_idx]) == 1:
                    del sub_clusters[c_idx]
                    self._sync_remove_subcluster(c_idx, super_clusters)
                else:
                    sub_clusters[c_idx].remove(i)

                # Compute PPM probability 
                weights = self.cluster_probabilities(i, sub_clusters)
                for c in range(len(sub_clusters)):
                    # Add similarity penaltt
                    penalty = self.compute_mahalanobis_penalty(sub_clusters[c], i)
                    weights[c] *= np.exp(-self.lambda_penalty * penalty)
                trans = random.choices(range(len(weights)), weights=weights)[0]

                if trans == len(sub_clusters):
                    sub_clusters.append([i])
                    # Create a new sub-cluster
                    new_sub_idx = len(sub_clusters) - 1
                    super_clusters.append([new_sub_idx])
                else:
                    sub_clusters[trans].append(i)

            # Level 2: PPMx on sub-clusters
            for j in range(len(sub_clusters)):
                
                # Find and remove j from its current supercluster
                s_idx = next(idx for idx, scl in enumerate(super_clusters) if j in scl)
                
                if len(super_clusters[s_idx]) == 1:
                    del super_clusters[s_idx]
                else:
                    super_clusters[s_idx].remove(j)
                
                # Compute the super-clusters probabilities
                probs = self._get_super_probabilities(j, super_clusters, sub_clusters)
                trans_super = random.choices(range(len(probs)), weights=probs)[0]
                
                if trans_super == len(super_clusters):
                    super_clusters.append([j])
                else:
                    super_clusters[trans_super].append(j)

            # Update history
            self.history.append(copy.deepcopy(sub_clusters))
            self.shistory.append(copy.deepcopy(super_clusters))
            # update_metrics
            self.update_metrics(metrics, sub_clusters)
            pbar.update(1)
            
        pbar.close()
        return self.history, self.shistory
    
    def binder_loss(self, clustering, similarity_matrix, alpha=1.0, beta=1.0):
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
        loss = 0.0
        n = similarity_matrix.shape[0]

        Z = np.zeros((n, n))
        for cluster in clustering:
            for i in cluster:
                for j in cluster:
                    Z[i, j] = 1.0

        for i in range(n):
            for j in range(i + 1, n):
                pij = similarity_matrix[i, j]
                zij = Z[i, j]
                loss += alpha * pij * (1 - zij) + beta * (1 - pij) * zij

        return loss

    
    def compute_similarity_matrix_level1(self, burnin=0):
        """
        Computes the similarity matrix based on the MCMC clustering history.

        Parameters:
            burn_in (int, optional): The number of initial iterations to discard before computing 
                                    the similarity matrix. Defaults to 0.

        Returns:
            np.ndarray: A (n_obs, n_obs) similarity matrix, where each entry (i, j) represents 
                        the proportion of MCMC samples in which observations i and j were clustered together.
        """
        n = self.n_obs
        S = np.zeros((n, n))
        
        samples = self.history[burnin:]
        T = len(samples)

        for clustering in samples:
            for cluster in clustering:
                for i in cluster:
                    for j in cluster:
                        S[i, j] += 1.0

        self.sub_similarity_matrix = S / T
        return self.sub_similarity_matrix
    
    def find_optimal_clustering(self, history, similarity_matrix, alpha=1.0, beta=1.0):
        """
        Finds the clustering that minimizes the Binder loss.

        Parameters:
            history (list of list): List of all sub-partitions
            simularity_matrix (np.ndarray): similarity matrix obtained from compute_similarity_matrix_level1
            alpha (float, optional): Weight for within-cluster disagreements. Defaults to 1.0.
            beta (float, optional): Weight for between-cluster disagreements. Defaults to 1.0.

        Returns:
            optimal_clustering (list of lists): The clustering configuration that minimizes the Binder loss.
            optimal_loss (float): The Binder loss value corresponding to the optimal clustering.
        """
        best_loss = np.inf
        best_clustering = None

        progress_bar = tqdm(
            total=len(history),
            desc="Point Estimate Progress",
            unit="step"
        )

        for clustering in history:
            loss = self.binder_loss(clustering, similarity_matrix, alpha, beta)
            if loss < best_loss:
                best_loss = loss
                best_clustering = clustering

            progress_bar.update(1)

        progress_bar.close()
        return best_clustering, best_loss


    def find_optimal_subclustering(self, burnin=0, alpha=1.0, beta=1.0):
        self.compute_similarity_matrix_level1(burnin)
        return self.find_optimal_clustering(
            self.history[burnin:],
            self.sub_similarity_matrix,
            alpha,
            beta
        )

    # --- METODI PER ALLINEARE I DUE LIVELLI ---

    def compute_super_similarity_on_fixed_subs(self, best_sub_clusters, burnin=0):
        """
        Calcola la matrice di similarità tra i Sub-Cluster OTTIMALI (fissi).
        
        Input:
            best_sub_clusters: lista di liste (es. [[0,1,5], [2,3], [4]]) trovata al Livello 1
        Output:
            S_matrix: Matrice KxK (dove K è il num di sub-clusters ottimali)
                      S[u, v] = Probabilità che il sub-cluster ottimale 'u' e 'v' 
                      siano nello stesso Super-Cluster.
        """
        K = len(best_sub_clusters)
        S = np.zeros((K, K))
        
        pivots = [cluster[0] for cluster in best_sub_clusters]
        
        samples_sub = self.history[burnin:]
        samples_super = self.shistory[burnin:]
        T = len(samples_sub)
        
        for t in range(T):
            # We recover the structure at this step
            current_subs = samples_sub[t]      # es. [[0,1,5], [2,3], [4]]
            current_supers = samples_super[t]  # es. [[0, 1], [2]]
            
            pivot_super_labels = []
            
            for p in pivots:
                # Find the index ‘idx’ such that p is in current_subs[idx]
                sub_idx = -1
                for idx, cluster in enumerate(current_subs):
                    if p in cluster:
                        sub_idx = idx
                        break
                
                # In which supercluster is this sub_idx located?
                super_label = -1
                if sub_idx != -1:
                    for s_label, s_cluster in enumerate(current_supers):
                        if sub_idx in s_cluster:
                            super_label = s_label
                            break
                
                pivot_super_labels.append(super_label)
            
            # Update S
            for u in range(K):
                for v in range(K):
                    if pivot_super_labels[u] == pivot_super_labels[v] and pivot_super_labels[u] != -1:
                        S[u, v] += 1.0
                        
        return S / T

    def get_hierarchical_structure(self, best_sub_clusters, burnin=0):
        """
        Restituisce la gerarchia completa coerente.
        
        Returns:
            best_super_structure: Una lista di liste di INDICI di best_sub_clusters.
            Es: [[0, 2], [1]] significa che il Sub-Cluster 0 e 2 (di best_sub_clusters) 
            stanno insieme fisicamente, mentre l'1 è solo.
        """
        # 1. Calculate the similarity matrix based on fixed groups
        S_super = self.compute_super_similarity_on_fixed_subs(best_sub_clusters, burnin)
        
        # 2.Find the optimal clustering of this reduced matrix.
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        # Convert similarities into distance
        dist = 1.0 - S_super
        np.fill_diagonal(dist, 0)
        
        # Se K = 1, ritorna banale
        if len(dist) <= 1:
            return [[0]]

        # Ward linkage on probabilities
        Z = linkage(squareform(dist), method='average')
        
        # Prune the tree to minimize the approximate Binder Loss (threshold 0.5 is standard for prob).
        labels = fcluster(Z, t=0.5, criterion='distance')
        
    
        num_groups = max(labels)
        structure = [[] for _ in range(num_groups)]
        for i, label in enumerate(labels):
            structure[label-1].append(i)
            
        return structure


