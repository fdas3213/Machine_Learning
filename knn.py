import numpy as np 
import scipy.io as sio
import sys

def get_data(filepath):
	dat = sio.loadmat(filepath)
	return dat['train'], dat['test']

def l2_norm(l1, l2):
	return np.linalg.norm(l2 - l1)

def majority_vote(dic):
	return max(dic.items(), key = lambda x: x[1])[0]

def calculate_dist(train, test, k):
	dist = [(i,l2_norm(test[1:], train[i,1:])) for i in range(len(train))]
	dist_ordered = sorted(dist, key = lambda x: x[1])[:k]
	labels = {v: 0 for v in np.arange(10, dtype = np.float32)}
	for idx, d in dist_ordered:
		labels[train[idx,0]] = labels.get(train[idx,0],0) + 1
	pred = majority_vote(labels)
	return 1 if pred == test[0] else 0

def calculate_accuracy(train, test, k):
	all_predictions = 0
	indices = np.random.choice(test.shape[0], 100, replace = False)
	for i in indices:
		all_predictions += calculate_dist(train, test[i],k)
	return all_predictions / 100

def find_best_k(train, test, k_list):
	pred = dict()
	for k in k_list:
		accuracy = calculate_accuracy(train, test, k)
		pred[k] = accuracy
	return pred

if __name__ == "__main__":
	k_list = [1,5,9,13]
	train, test = get_data(sys.argv[1])
	result = find_best_k(train, test, k_list)
	print(result)
