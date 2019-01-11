import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#define batch_size, input dimension, hidden dimension, and output dimension
N, D_in, D_H, D_out = 32, 1000, 100, 10

#create random data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

#initialize weight matrices
w1 = np.random.rand(D_in, D_H)
w2 = np.random.rand(D_H, D_out)

#define learning rate, num of epochs
lr = 1e-5
NUM_EPOCH = 1000
MOMENTUM = 0.9
V_1, V_2 = np.zeros(w1.shape), np.zeros(w2.shape)
loss = list()

for epoch in tqdm(range(NUM_EPOCH)):
	h = x.dot(w1)
	h_relu = np.maximum(h, 0)
	pred = h_relu.dot(w2)

	#compute loss
	l = np.square(y - pred).sum()/N
	loss.append(l)
	if (epoch+1) % 100 == 0:
		print("Epoch: {}, loss: {:.3f}".format(epoch, l))

	#backpropagates using SGD with momentum
	grad_pred = 2*(pred - y)
	grad_w2 = h_relu.T.dot(grad_pred)
	grad_h_relu = grad_pred.dot(w2.T)
	grad_h = grad_h_relu.copy()
	grad_h[h < 0] = 0
	grad_w1 = x.T.dot(grad_h)

	#update parameters
	V_1 = MOMENTUM * V_1 + (1- MOMENTUM) * grad_w1
	V_2 = MOMENTUM * V_2 + (1- MOMENTUM) * grad_w2
	w1 -= lr * V_1
	w2 -= lr * V_2


plt.plot(list(range(len(loss))), loss)
plt.show()