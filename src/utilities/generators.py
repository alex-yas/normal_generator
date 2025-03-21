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
        
        if any(np.linalg.eig(correlation_matrix)[0] < 0):
            raise Exception("Invalid correlation matrix, " +
                "it should be positive semi-definite") 

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

            correlated_observation = self.correlation_matrix.dot(observation)
            yield correlated_observation