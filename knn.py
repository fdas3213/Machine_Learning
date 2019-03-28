import numpy as np 
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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

