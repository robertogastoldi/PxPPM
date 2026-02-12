import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

def plot_clusters(data, optimal_clustering, title, xlabel="X1", ylabel="X2"):
    """
    Plots clusters based on the given clustering information.

    Parameters:
        data (np.ndarray): The dataset, where rows are observations and columns are features.
        optimal_clustering (list of lists): The clustering information as a list of lists, 
                                            where each sublist contains indices of points in a cluster.
        title (str): The title of the graph.
        xlabel (str, optional): The label for the x-axis. Defaults to "X1".
        ylabel (str, optional): The label for the y-axis. Defaults to "X2".

    Returns:
        None: Displays the scatter plot of the clusters.
    """

    colors = plt.cm.rainbow(np.linspace(0, 1, len(optimal_clustering)))
    plt.figure(figsize=(8, 6))

    for cluster_id, observations in enumerate(optimal_clustering):
        cluster_data = data[observations]
        plt.scatter(cluster_data[:, 0], 
                    cluster_data[:, 1], 
                    label=f"Cluster {cluster_id}", 
                    color=colors[cluster_id],
                    s=100,
                    alpha=0.8,
                    edgecolor="k")

    # Add labels, legend, and grid
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()

def plot_superclusters(Y, optimal_sub, optimal_super, title="Sub-clusters and Super-clusters", xlabel="Y1", ylabel="Y2"):
    """
    Plot 2D data showing:
    - sub-clusters via point colors
    - super-clusters via convex hull areas

    Parameters
    ----------
    Y : np.ndarray (n_obs, 2)
        Spatial coordinates.
    optimal_sub : list of lists
        Sub-clustering (indices of observations).
    optimal_super : list of lists
        Super-clustering (indices of sub-clusters).
    """

    plt.figure(figsize=(9, 7))

    # --- 1. Colori per sub-cluster ---
    sub_colors = plt.cm.tab10(np.linspace(0, 1, len(optimal_sub)))

    # --- 2. Plot punti (sub-cluster) ---
    for j, sub in enumerate(optimal_sub):
        pts = Y[sub]
        plt.scatter(
            pts[:, 0],
            pts[:, 1],
            color=sub_colors[j],
            edgecolor="k",
            s=70,
            alpha=0.9,
            label=f"Sub-cluster {j}"
        )

    # --- 3. Disegno super-cluster (aree) ---
    super_colors = plt.cm.Set2(np.linspace(0, 1, len(optimal_super)))

    for k, super_k in enumerate(optimal_super):

        # Recupera TUTTI i punti appartenenti ai sub-cluster del super-cluster k
        indices = []
        for sub_idx in super_k:
            indices.extend(optimal_sub[sub_idx])

        points = Y[indices]

        # Convex hull solo se ha senso
        if len(points) >= 3:
            hull = ConvexHull(points)
            hull_pts = points[hull.vertices]

            plt.fill(
                hull_pts[:, 0],
                hull_pts[:, 1],
                color=super_colors[k],
                alpha=0.25,
                label=f"Super-cluster {k}"
            )

        else:
            # Caso degenerato: pochi punti
            plt.scatter(
                points[:, 0],
                points[:, 1],
                color=super_colors[k],
                s=200,
                alpha=0.3,
                marker="o"
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha = 0.3)
    plt.legend(loc="best", fontsize=9)
    plt.show()

