import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

class PrincipleComponentAnalysis:
    def __init__(self, n_components):
        self.n_componenets = n_components

    def fit(self, X):
        self.cov_mat = np.dot(X.T, X)
        lam, mu = np.linalg.eig(self.cov_mat)
        self.outX = np.zeros((X.shape[0], self.n_componenets))
        transformed_matrix = np.dot(X, mu.T)
        self.outX[:,:self.n_componenets] = transformed_matrix[:,:self.n_componenets]
        return self.outX

if __name__ == "__main__":
    X = load_iris()['data']
    p = PrincipleComponentAnalysis(n_components=2)
    X_t_1 = p.fit(X)
    plt.scatter(X_t_1[:,0], X_t_1[:,1])
    plt.show()