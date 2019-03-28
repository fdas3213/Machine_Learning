import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from numpy.linalg import norm

class KMeans:
	#implement kmeans 
	def __init__(self, num_clusters):
		self.n_cluster = num_clusters

	def l2_dist(self, x, y):
		#use euclidean distance
		return norm(x-y, axis = 1)

	def initCenter(self, x):
		#initialize class centers
		N, D =x.shape
		return x[np.random.randint(0, N, self.n_cluster)]

	def fit(self, x):
		N, D = x.shape
		self.clusters = np.zeros(N)
		nonstop = True
		#step 1: initialize cluster centers
		self.init_means = self.initCenter(x)
		#step 2: iteratively update class center
		while nonstop:
			nonstop = False
			for i in range(N):
				dist = self.l2_dist(x[i], self.init_means)
				cluster_idx = np.argmin(dist)
				if self.clusters[i] != cluster_idx:
					nonstop = True
					self.clusters[i] = cluster_idx
			#update cluster centroids
			for k in range(self.n_cluster):
				self.init_means[k] = np.mean(x[self.clusters == k], axis = 0)

	def draw(self, data):
		plt.scatter(data[self.clusters == 0, 0], data[self.clusters == 0, 1], c = 'red')
		plt.scatter(data[self.clusters == 1, 0], data[self.clusters == 1, 1], c = 'blue')
		plt.scatter(data[self.clusters == 2, 0], data[self.clusters == 2, 1], c = 'green')
		plt.scatter(self.init_means[:,0], self.init_means[:,1], c = 'yellow')
		plt.legend(['cluster_0', 'cluster_1', 'cluster_2', 'center'])
		plt.show()

if __name__ == "__main__":
	#load data and shuffle indices
	x = load_iris()['data']
	shuffle_indices = np.random.permutation(len(x))
	x = x[shuffle_indices]
	#fit KMeans model
	clf = KMeans(3)
	clf.fit(x)
	#draw the plot
	clf.draw(x)

