import unittest

import numpy as np

from utilities import (compute_mean, compute_deviation, 
                       compute_correlation_matrix)

class TestEmpiricalCharacteristics(unittest.TestCase):

    def test_compute_mean(self):
        observations = np.array([[1, 2, 3, 4, 5]])
        
        self.assertEqual(compute_mean(observations), np.array([3]))

    def test_compute_deviation(self):
        observations = np.array([[2, 4, 4, 4, 5, 5, 7, 9]])

        self.assertEqual(compute_deviation(observations), np.array([2]))

    
    def test_compute_correlation_matrix(self):
        observations = np.array([
            [1, 2, 3],
            [-1, -2, -3]
        ])

        self.assertTrue((compute_correlation_matrix(observations) == np.array([[1, -1], [-1, 1]])).all())


if __name__ == '__main__':
    unittest.main()