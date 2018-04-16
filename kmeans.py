import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets
from numpy.linalg import norm

def load_data():
	iris = datasets.load_iris()
	x = iris['data']
	y = iris['target']
	shuffle_indices = np.random.permutation(len(x))
	return x[shuffle_indices], y[shuffle_indices]

def l2_norm(x,y):
	return norm(x-y)

def initCenter(data, num_cluster):
	numSamples, dim = data.shape
	return data[np.random.randint(0, numSamples, num_cluster)]

def k_means(num_cluster, data):
	numSamples, dim = data.shape
	clusters = np.zeros(numSamples)
	nonstop = True
	#step 1: initialize cluster centers
	init_means = initCenter(data, num_cluster)

	while nonstop:
		nonstop = False
		#for each sample
		for i in range(numSamples):
			#find the closest centroid
			dist = norm(data[i] - init_means, axis = 1)
			cluster_idx = np.argmin(dist)
			if clusters[i] != cluster_idx:
				nonstop = True
				clusters[i] = cluster_idx

		#update centroids
		for k in range(num_cluster):
			init_means[k] = np.mean(data[clusters == k], axis = 0)
			
	return clusters, init_means

def draw(num_cluster, data):
	clusters, mean = k_means(num_cluster, data)
	plt.scatter(data[clusters == 0, 0], data[clusters == 0, 1], c = 'red')
	plt.scatter(data[clusters == 1, 0], data[clusters == 1, 1], c = 'blue')
	plt.scatter(data[clusters == 2, 0], data[clusters == 2, 1], c = 'green')
	plt.scatter(mean[:,0], mean[:,1], c = 'yellow')
	plt.show()
	
if __name__ == "__main__":
	x, y = load_data()
	draw(3, x)
	#c, m = k_means(3,x)

