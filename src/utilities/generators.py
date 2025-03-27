import numpy as np
from typing import Iterator, List


class NormalGenerator():
    def __init__(self, mean: np.array, variance: np.array, dimension: int):
        self.mean = mean
        self.variance = variance
        self.dimension = dimension
    
    def generate(self) -> Iterator[List[float]]:
        """ Generator for normal distribution

        Yields:
            Iterator[List[float]]: observations of normal distribution
        """
        while True:
            observation = [
                np.random.normal(loc=self.mean[i], scale=self.variance[i], 
                                 size=1).item()
                for i in range(self.dimension)
            ]
            
            yield observation


class CorrelatedNormalGenerator(NormalGenerator):
    def __init__(self, mean: np.array, variance: np.array, dimension: int, 
                 correlation_matrix: np.array):
        super(CorrelatedNormalGenerator, self).__init__(mean, variance, 
                                                        dimension)
        
        eigen_values = np.linalg.eig(correlation_matrix)[0]
        
        if any(eigen_values < 0):
            correlation_matrix = self._compute_similar_matrix(
                correlation_matrix
            )

            print(
                f"Invalid correlation matrix, " + 
                f"it should be positive semi-definite. " + 
                f"Using similar correlation matrix: {correlation_matrix}"
            ) 

        self.correlation_matrix = np.linalg.cholesky(
            np.array(correlation_matrix)
        )
    
    def generate(self) -> Iterator[List[float]]:
        """ Generator for correlated normal distribution

        Yields:
            Iterator[List[float]]: observations of normal distribution
        """
        while True:
            observation = [
                np.random.normal(loc=self.mean[i], scale=self.variance[i], 
                                 size=1).item()
                for i in range(self.dimension)
            ]

            correlated_observation = self.correlation_matrix @ observation

            yield correlated_observation

    def _compute_similar_matrix(self, matrix: np.array) -> np.array:
        eigen_values, eigen_vectors = np.linalg.eig(matrix)
        eigen_values[eigen_values < 0] = 1e-3

        inversed_eigen_vectors = np.linalg.inv(eigen_vectors)
        eigen_diagonal = np.diag(eigen_values)
        
        similar_matrix = eigen_vectors @ eigen_diagonal @ inversed_eigen_vectors
        
        return similar_matrix