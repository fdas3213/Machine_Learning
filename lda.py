#!/usr/bin/python3
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

class LDA:
	def __init__(self):
		pass

	def fit(self, x, y):
		self.train_x = x
		self.train_y = y
		self.size, dimension = self.train_x.shape
		self.n_classes = np.unique(y).shape[0]
		self.means = np.zeros((self.n_classes, dimension))
		self.priors = np.zeros(self.n_classes)
		for i in range(self.n_classes):
			self.means[i] = np.mean(self.train_x[self.train_y==i])
			self.priors[i] = len(self.train_y[self.train_y==i])/self.size
		self.cov_mat = np.zeros((dimension, dimension))
		for i in range(self.size):
			vec = self.train_x[i]-self.means[self.train_y[i]]
			self.cov_mat += np.outer(vec.T, vec)
		self.cov_mat /= self.size

	def gaussian(self, x, mu, sigma, prior):
		sec = np.dot(x-mu, np.linalg.inv(sigma)).dot((x-mu).T)
		return np.log(prior) - 1/2*sec

	def predict(self, x_test):
		preds = np.zeros(len(x_test))
		for idx in range(len(x_test)):
			tmp = np.zeros(self.n_classes)
			for k in range(self.n_classes):
				tmp[k] = self.gaussian(x_test[idx], self.means[k], self.cov_mat, self.priors[k])
			preds[idx] = np.argmax(tmp)
		return preds

if __name__ == "__main__":
	#use iris dataset to test
	x, y = load_iris()['data'], load_iris()['target']
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
	clf = LDA()
	clf.fit(x_train, y_train)
	preds = clf.predict(x_test)
	print("accuracy = {:.3f}".format(accuracy_score(preds, y_test)))
	## compare with sklearn
	sk_clf = LinearDiscriminantAnalysis()
	sk_clf.fit(x_train, y_train)
	sk_preds = sk_clf.predict(x_test)
	print("sklearn accuracy = {:.3f}".format(accuracy_score(sk_preds, y_test)))