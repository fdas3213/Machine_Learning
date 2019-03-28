import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class LinearRegre:
	def __init__(self, num_epochs, lr, bias = True):
		self.num_epochs = num_epochs
		self.learning_rate = lr
		self.bias = bias

	def get_loss(self, x, y,w,b):
		return np.square(x.dot(w)+b-y)

	def cal_grad(self, w,x,y, b):
		return 2*x*(np.dot(x,w)+b - y), 2*(np.dot(x,w)+b-y)

	def fit(self, x, y):
		#perform SGD
		self.w = np.random.uniform(-1,1, size = x.shape[1])
		self.b = 0
		self.total_loss = list()
		data_size = x.shape[0]
		for i in range(self.num_epochs):
			iter_loss = 0
			for idx in range(data_size):
				w_grad, b_grad = self.cal_grad(self.w, x[idx], y[idx], self.b)
				iter_loss += self.get_loss(x[idx], y[idx], self.w, self.b)
				self.w -= self.learning_rate * w_grad
				self.b -= self.learning_rate * b_grad
			if i % 1000 == 0:
				print("Iteration: {}, loss: {:.4f}".format(i, iter_loss/data_size))
			self.total_loss.append(iter_loss/data_size)

	def predict(self, x_test):
		return x_test.dot(self.w) + self.b

if __name__ == "__main__":
	#test using diabetes dataset
	data = datasets.load_diabetes()
	x, y = data['data'], data['target']
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
	clf = LinearRegre(10000, 0.01)
	clf.fit(x_train, y_train)
	preds = clf.predict(x_test)
	print("self implemented mean squared error is: {:.4f}".format(mean_squared_error(preds, y_test)))
	#compared with Sklearn implementation
	lr = LinearRegression()
	lr.fit(x_train, y_train)
	sk_preds = lr.predict(x_test)
	print("Sklearn LinearRegression mean squared error is: {:.4f}".format(mean_squared_error(sk_preds, y_test)))