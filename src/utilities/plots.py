import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from typing import List


def plot_density(data: np.array, title: str, labels: List[str], 
                 colors: List[str]) -> None:
    """Plots empirical density of distribution

    Args:
        data (np.array): observations
        title (str): title of the plot
        labels (List[str]): labels for each variable
        colors (List[str]): colors for each variable
    """
    for i in range(data.shape[0]):
        density = gaussian_kde(data[i])
        xs = np.linspace(-5, 11, data.shape[1])
        density._compute_covariance()
        plt.plot(xs, density(xs), color=colors[i], label=labels[i])
        plt.title(title)
        plt.legend()

    plt.show()


def plot_heatmap(data: np.array, title: str, x_labels: List[str], 
                 y_labels: List[str]) -> None:
    """ Plots heatmap by specific data

    Args:
        data (np.array): data to plot
        title (str): title of the plot
        x_labels (List[str]): labels for X axis
        y_labels (List[str]): labels for Y axis
    """
    fig, ax = plt.subplots()
    ax.imshow(data)
    
    ax.set_xticks(range(len(x_labels)), labels=x_labels)
    ax.set_yticks(range(len(y_labels)), labels=y_labels)

    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            ax.text(j, i, round(data[i, j], 3), ha="center", va="center", color="r")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()