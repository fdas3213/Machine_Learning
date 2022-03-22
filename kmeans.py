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



class KMeansV1:
	# new version
    def __init__(self, n_clusters, method='l2'):
        self.n_clusters = n_clusters
        self.X = None
        if method not in ['l2','cosine']:
            raise ValueError("Please use either L2-distance or Cosine distance")
        self.method = method
    
    def _calc_l2(self, x):
        return np.sqrt(np.sum(np.square(x)))
        #return np.sqrt(np.sum(np.square(x), axis=1)) for matrix computation
    
    def _calc_dist(self, a, b):
        if self.method=='l2':
            return self._calc_l2(a-b)
        elif self.method=='cosine':
            return np.dot(a,b)/(self._calc_l2(a)*self._calc_l2(b))
    
    def fit(self, x):
        self.X = x
        N, d = x.shape
        #initialize cluster labels for each data point
        self.cluster_labels = np.zeros(N)
        #initialize cluster centers
        self.cluster_centers = [self.X[random.randint(0,N-1)] for _ in range(self.n_clusters)]
        #iteratively update cluster label for each data point
        stop = False
        while not stop:
            stop = True
            for index, x_tr in enumerate(x):
                #assign x_tr to the cluser which has the closest distance to the cluster center
                cur_cluster = np.argmin([self._calc_dist(x_tr, centroid) for centroid in self.cluster_centers])
                prev_cluster = self.cluster_labels[index]
                # does not converge
                if cur_cluster != prev_cluster:
                    self.cluster_labels[index] = cur_cluster
                    stop = False
            
            #update cluster centroids
            for i in range(self.n_clusters):
                self.cluster_centers[i] = np.mean(self.X[self.cluster_labels==i], axis=0)
                
    def predict(self, x_test):
        preds = []
        for x_te in x_test:
            cluster_label = np.argmin([self._calc_dist(x_te, centroid) for centroid in self.cluster_centers])
            preds.append(cluster_label)
        
        return preds 

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

