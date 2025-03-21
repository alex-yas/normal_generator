import numpy as np
    
def compute_mean(observations: np.array) -> np.array:
    """Computes means of n-D variable observations

    Args:
        observations (2d np.array): variable observations

    Returns:
        np.array: compute mean for each varible dimension
    """
    means = np.array([
        np.mean(variable_observations).item()
        for variable_observations in observations
    ])

    return means

def compute_deviation(observations: np.array) -> np.array:
    """Computes standard deviations of n-D variable observations

    Args:
        observations (2d np.array): variable observations

    Returns:
        np.array: compute mean for each varible dimension
    """
    deviations = np.array([
        np.std(variable_observations).item()
        for variable_observations in observations
    ])

    return deviations

def compute_correlation_matrix(observations: np.array) -> np.array:
    """Computes correlation matrix

    Args:
        observations (2d np.array): variable observations

    Returns:
        2d np.array: correlation matrix
    """
    return np.corrcoef(observations)
