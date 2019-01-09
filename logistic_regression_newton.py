import sys
import numpy as np
import matplotlib.pyplot as plt

def read_feature(filepath):
	with open(filepath) as f:
		dat = [[float(val) for val in line.split()] for line in f]

	return np.column_stack((np.ones(len(dat)),np.array(dat)))

def read_label(filepath):
	with open(filepath) as f:
		dat = [float(line.rstrip()) for line in f ]
	return np.array(dat)

def hessian(x,y,w):
	W = np.zeros((len(x),len(x)))
	for idx in range(len(x)):
		h_x = 1 / (1+np.exp(-w.T.dot(x[idx])))
		W[idx,idx] = h_x * (1-h_x)
	H = x.T.dot(W).dot(x)
	return np.linalg.inv(H)

def gradient(x,y,w):
	W = np.zeros(len(x))
	for idx in range(len(x)):
		h_x = 1 / (1+np.exp(-w.T.dot(x[idx])))
		W[idx] = h_x - y[idx]
	G = x.T.dot(W)
	return G

def cross_entropy_loss(x,y,w):
	total_loss = 0
	for idx in range(len(x)):
		h_x = 1 / (1+np.exp(-w.T.dot(x[idx])))
		l = -y[idx]*np.log(h_x) - (1-y[idx])*np.log(1-h_x)
		total_loss += l
	return total_loss

def newton_method(feature,label,epsilon):
	w = np.zeros(3)
	l = cross_entropy_loss(feature, label,w)
	delta_loss = np.Infinity
	i = 0
	error = list()
	while delta_loss > epsilon:
		g = gradient(feature,label, w)
		h_inv = hessian(feature, label, w)
		delta = h_inv.dot(g)
		w -= delta
		l_new = cross_entropy_loss(feature, label,w)
		delta_loss = np.abs(l_new - l)
		error.append(l)
		l = l_new
		i += 1
		
	return i, w, error

def plot(iteration, error, train = True):
	plt.plot([i for i in range(iteration)], error)
	plt.xlabel("Number of iteration")
	plt.ylabel("Total loss")
	if train:
		plt.title("Training error vs. number of iteration")
	else:
		plt.title("Test error vs. number of iteration")
	plt.show()

if __name__ == "__main__":
	#first command line argument is training/test feature data, second command line argument is training/test label data
	feature = read_feature(sys.argv[1])
	label = read_label(sys.argv[2])
	iteration, coeff, errors = newton_method(feature, label, 1e-8)
	#print("iteration: {}, coeff, {}".format(iteration, coeff))
	plot(iteration, errors, True)

