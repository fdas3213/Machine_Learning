import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt

class LinearSGD:
	def __init__(self, learning_rate = 0.01, num_iter = 5):
		self.lr = learning_rate
		self.num_iter = num_iter

	def __square_loss(self, y, y_hat):
		return np.square(y - y_hat)

	def __predict(self, x, a,b):
		return x*a + b

	def __grad(self, x, y, a, b):
		return np.array([-2*x*(y-a*x-b), -2*(y-a*x-b)])

	def fit(self, X, y, gamma = 0.9):
		#add momentum to SGD
		#weight and velocity initialization
		self.weights = np.random.rand(2)
		self.v_t = np.zeros(2)

		#record loss
		self.loss = list()
		dim = len(X)
		for n_iter in tqdm(range(self.num_iter)):
			iter_loss = 0
			p = np.random.permutation(dim)
			X, y = X[p], y[p]
			for i in range(len(x)):
				pred = self.__predict(X[i], *self.weights)
				l = self.__square_loss(y[i], pred)
				iter_loss += l
				self.v_t = gamma * self.v_t + (1 - gamma) * self.__grad(X[i], y[i], *self.weights)
				self.weights -= self.lr * self.v_t
			self.loss.append(iter_loss/dim)

	def plot_loss(self):
		plt.plot(list(range(len(self.loss))), self.loss)
		plt.title("error by epoch")
		plt.show()
		

if __name__ == "__main__":
	x = np.array([14, 86, 28, 51, 28, 29, 72, 62, 84, 15, 42, 62, 47, 35,  9, 38, 44,
       99, 13, 21, 28, 20, 8,64,99,70,27,17,8])
	y = np.array([ 58, 202,  86, 132,  86,  88, 174, 154, 198,  60, 114, 154, 124,
       100,  48, 106, 118, 228,  56,  72,  86, 70,46,158,228,170,84,64,46])
	model = LinearSGD(learning_rate = 0.0001, num_iter = 50)
	model.fit(x, y)
	model.plot_loss()
