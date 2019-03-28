import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LogisticRegression:
	##implements Logistic Regression using SGD
	def __init__(self, learning_rate, epsilon = 1e-2, bias = True):
		self.lr = learning_rate
		self.epsilon = epsilon
		self.bias = bias

	def __sigmoid(self, x, w):
		return 1/(1+np.exp(-x.dot(w)))

	def __CELoss(self, x, y, w):
		pred = self.__sigmoid(x, w)
		return -y*np.log(pred) - (1-y)*np.log(1-pred)

	def __gradient(self, x, y, w):
		pred = self.__sigmoid(x, w)
		return (pred - y)*x

	def fit(self, x,y):
		#perform SGD
		data_size = len(x)
		self.dimension = x.shape[1]
		if self.bias:
			x_train = x.copy()
			x = np.zeros((data_size, self.dimension+1))
			x[:, :-1] = x_train
		self.w = np.random.uniform(-1, 1, size = self.dimension+1)
		self.total_loss = list()
		pre_loss = 1
		delta_loss = np.Infinity
		while delta_loss > self.epsilon:
			iter_loss = 0
			for idx in range(data_size):
				w_grad = self.__gradient(x[idx], y[idx], self.w)
				iter_loss += self.__CELoss(x[idx], y[idx], self.w)
				self.w -= self.lr * w_grad
			delta_loss = np.abs(iter_loss - pre_loss)/pre_loss
			pre_loss = iter_loss
			self.total_loss.append(iter_loss)

	def predict(self, x_test):
		if self.bias:
			x = x_test.copy()
			x_test = np.zeros((len(x_test), self.dimension + 1))
			x_test[:, :-1] = x
		self.probs = x_test.dot(self.w)
		self.pred = self.probs.copy()
		self.pred[self.pred >= 0.5] = 1
		self.pred[self.pred < 0.5] = 0
		return self.pred

if __name__ == "__main__":
	#test using breast cancer dataset
	data = load_breast_cancer()
	x, y = data['data'], data['target']
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
	clf = LogisticRegression(0.01, epsilon = 1e-8)
	clf.fit(x_train, y_train)
	preds = clf.predict(x_test)
	print("self implemented LogisticRegression accuracy_score: {:.4f}".format(accuracy_score(preds, y_test)))
