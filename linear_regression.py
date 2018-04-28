import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

def load_data():
	#add a 0 column to X 
	data = datasets.load_diabetes()
	x, y = data['data'],data['target']
	return x[:-20],x[-20:], y[:-20],y[-20:]

def loss(x, y, w):
	pred = x.dot(w)
	l = pred - y
	return np.sum(l**2) / (2*x.shape[0])

def cal_grad(w, x, y):
	l = np.dot(x, w) - y
	assert l.shape == y.shape
	grad = np.dot(x.T, l) / x.shape[0]
	return grad

def gradient_descent(iteration, learning_rate, x, y):
	#perform gradient descent for parameter w 
	w = np.random.rand(x.shape[1])
	total_loss = []
	for i in range(iteration):
		w_grad = cal_grad(w, x,y)
		l = loss(x,y,w)
		if i % 1000 == 0:
			print("Iteration: {} has loss: {}".format(i, l))
		w -= learning_rate * w_grad
		total_loss.append(l)

	return w,total_loss

def fit(x, w):
	return x.dot(w)


if __name__ == "__main__":
	x_train, x_test, y_train, y_test = load_data()
	w, l = gradient_descent(10000, 0.1, x_train,y_train)
	y_hat = fit(x_test, w)
	plt.scatter(x_test[:,2], y_test, color = 'black')
	plt.plot(x_test[:,2], y_hat,color = 'blue', linewidth = 3)
	plt.show()

