import numpy as np 
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from collections import Counter
import sys

class KNN:
	def __init__(self, k = 3):
		self.k = k

	def l2_norm(self, l1, l2):
		return np.linalg.norm(l2-l1)

	def majority_vote(self, dic):
		return max(dic.items(), key = lambda x: x[1])[0]

	def fit(self, x, y):
		self.train_x = x
		self.train_y = y

	def predict(self, x_test):
		data_size = len(self.train_x)
		preds = list()
		for test_idx in range(len(x_test)):
			dist = [(i, self.l2_norm(x_test[test_idx], self.train_x[i])) for i in range(data_size)]
			dist_ordered = sorted(dist, key = lambda x: x[1])[:self.k]
			labels = dict()
			for idx, d in dist_ordered:
				labels[self.train_y[idx]] = labels.get(self.train_y[idx], 0) + 1
			preds.append(self.majority_vote(labels))
		return preds


class KNN:
	#new implementation
    def __init__(self, k, dist_method='l2'):
        self.k = k
        self.dist_method=dist_method
        if dist_method not in ['l2', 'cosine']:
            raise ValueError("Please use either L2-distance or Cosine distance")
        # initialize training data to None in the Constructor
        self.x = None
        self.y = None
    
    def l2_norm(self, a):
        return np.sqrt(np.sum(np.square(a)))
    
    def calc_dist(self, a, b):
        if self.dist_method=='l2':
            return self.l2_norm(a-b)
        elif self.dist_method=='cosine':
            return np.dot(a,b)/(self.l2_norm(a) * self.l2_norm(b))
        
    def fit(self, x, y):
        self.x = x
        self.y = y
    
    def find_majority(self, labels):
        counts = Counter(labels)
        output = counts.most_common()[0][0]
        return output
    
    def predict(self, x_test):
        #find the k nearest neighbors
        preds = []
        for x_t in x_test:
            distances = [(self.calc_dist(x_train, x_t), y_train) for x_train, y_train in zip(self.x, self.y)]
            neighbor_labels = [y_tr for x_tr, y_tr in sorted(distances, key=lambda x:x[0])[:self.k]]
            preds.append(self.find_majority(neighbor_labels))
        
        return preds

if __name__ == "__main__":
	#use iris dataset to test
	x, y = load_iris()['data'], load_iris()['target']
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
	#find best k from this list [3,5,7,11]
	for K in [3,5,7,11]:
		clf = KNN(K)
		clf.fit(x_train, y_train)
		preds = clf.predict(x_test)
		print("k = {}, accuracy = {:.3f}".format(K, accuracy_score(preds, y_test)))

