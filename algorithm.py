""" Kmeans for pokemon dataset """

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def initialize_clusters(K, data, m):
    """ Choose randomly k clusters from dataset examples

    Parameters:
        K (int): Clusters quantity
        m (int): Number of training examples

    Returns:
        (list(n-dimensioanl-vector)): Clusters generated
    """
    return [data.iloc[index].values for index in
            np.random.randint(low=0, high=m-1, size=K)]


def J(μk, xi):
    """ Optimization objective

    Parameters:
        μk (vector): Cluster centroid k
        xi (vector): Training example i

    Returns:
        (list(n-dimensioanl-vector)): Clusters generated
    """
    return np.linalg.norm(xi.values[:2] - μk[:2])


def assignment(data, μ):
    """ Assign training examples to centroid

    Parameters:
        data (dataframe): Training examples
        μ (list(vector)): Clusters

    Returns:
        data (dataframe): Labeled to cluster
    """
    for index, xi in data.iterrows():
        min = sorted([
            (J(μk, xi), i)
            for i, μk in enumerate(μ)], reverse=False,
            key=lambda pair: pair[0])[0]
        data.iloc[index]["Distance"] = min[0]
        data.iloc[index]["Centroid"] = min[-1]

    return data


def update_clusters(data, μ):
    """ Moves clusters to mean of labeled data

    Parameters:
        data (dataframe): Training examples
        μ (list(vector)): Clusters

    Returns:
        μ (list(vector)): New clusters
        old_μ (list(vector)): Original clusters
    """
    old_μ = μ.copy()
    for index, _ in enumerate(μ):
        μ[index] = data.loc[data["Centroid"] == index].mean(axis=0).values

    return μ, old_μ


def plot_result(data):
    """ Plots kmeans

    Parameters:
        data (dataframe): Training examples

    """
    data["Centroid"] = data["Centroid"].apply(
        lambda x: "r" if x == 1 else "g" if x == 2  else "b")
    data.plot.scatter(x="Attack", y="Defense", c=data["Centroid"])
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("pokemon_data.csv")
    K = 3
    m = len(data.index)

    data = data.drop(columns=[
        "#", "Name", "Type 1", "Type 2", "Legendary", #"Generation"])
        "Sp. Atk", "Sp. Def", "Speed", "Generation", "HP"])
    data["Distance"] = -1
    data["Centroid"] = -1

    μ = initialize_clusters(K, data, m)
    while True:
        data = assignment(data, μ)
        μ, old_μ = update_clusters(data, μ)

        if ((np.array(μ) == np.array(old_μ)).all()):
            break

    plot_result(data)
