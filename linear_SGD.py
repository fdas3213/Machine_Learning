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

	def fit(self, X, y):
		#weight initialization
		self.w_1 = 1
		self.w_2 = 1

		#record loss
		self.loss = list()
		dim = len(X)
		for n_iter in tqdm(range(self.num_iter)):
			iter_loss = 0
			p = np.random.permutation(dim)
			X, y = X[p], y[p]
			for i in range(len(x)):
				pred = self.__predict(X[i], self.w_1, self.w_2)
				l = self.__square_loss(y[i], pred)
				iter_loss += l
				self.w_1 -= self.lr* (-2*X[i]*(y[i] - self.w_1 *X[i]-self.w_2))
				self.w_2 -= self.lr* (-2*(y[i] - self.w_1*X[i]-self.w_2))
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
