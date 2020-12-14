import copy
import torch
import random
from torch import nn
import numpy as np
from collections import defaultdict

def split(n_samples, n_participants, train_dataset=None, mode='uniform'):
	random.seed(1234)
	indices = np.arange(len(train_dataset))
	random.shuffle(indices)
	indices = indices[:n_samples]
	
	if mode == 'powerlaw':
		import math
		from scipy.stats import powerlaw

		# alpha = 1.65911332899
		alpha = 1
		mean_size = len(indices)//n_participants # middle value
		beta=np.linspace(powerlaw.ppf(0.01, alpha),powerlaw.ppf(0.99, alpha), n_participants)

		participant_set_size=list(map(math.ceil, beta/sum(beta)*mean_size*n_participants))
		participant_indices={}
		accessed=0
		for nid in range(n_participants):
			participant_indices[nid] = indices[accessed:accessed+participant_set_size[nid]]
			accessed=accessed+participant_set_size[nid]

		train_indices = [v for k,v in participant_indices.items()]

	elif mode == 'classimbalance':
		n_classes = len(train_dataset.classes)
		data_indices = [(train_dataset.targets == class_id).nonzero().view(-1).tolist() for class_id in range(n_classes)]
		class_sizes = np.linspace(1, n_classes, n_participants, dtype='int')
		mean_size = n_samples // n_participants # for mnist mean_size = 600

		participant_indices = defaultdict(list)
		for participant_id, class_sz in enumerate(class_sizes):	
			classes = range(class_sz) # can customize classes for each participant rather than just listing
			each_class_id_size = mean_size // class_sz
			for i, class_id in enumerate(classes):
				selected_indices = data_indices[class_id][:each_class_id_size]
				data_indices[class_id] = data_indices[class_id][each_class_id_size:]
				participant_indices[participant_id].extend(selected_indices)

				# top up to make sure all participants have the same number of samples
				if i == len(classes) - 1 and len(participant_indices[participant_id]) < mean_size:
					extra_needed = mean_size - len(participant_indices[participant_id])
					participant_indices[participant_id].extend(data_indices[class_id][:extra_needed])
					data_indices[class_id] = data_indices[class_id][extra_needed:]
		train_indices = [participant_index_list for participant_id, participant_index_list in participant_indices.items()] 

	elif mode == 'disjointclasses':

		# each participant has some number partitioned classes of randomly selected examples
		# Works for a 5-participant case for MNIST or CIFAR10
		all_classes = np.arange(len(train_dataset.classes))
		data_indices = [(train_dataset.targets == class_id).nonzero().view(-1).tolist() for class_id in all_classes]
		mean_size = n_samples // n_participants

		# random.seed(1234)
		class_sz = 2
		multiply_by = class_sz * n_participants // len(all_classes)
		cls_splits = [cls.tolist() for cls in np.array_split(all_classes, np.ceil( 1.0 * len(train_dataset.classes) / class_sz)  )] 
		print("Using disjoint classes and partitioning the dataset to {} participants with each having {} classes.".format(n_participants, class_sz))

		clses = sorted([ cls for _, cls in zip(range(n_participants), repeater(cls_splits)) ])

		participant_indices = defaultdict(list)
		for participant_id, classes in enumerate(clses):
			print("participant id: {} is getting {} classes.".format(participant_id, classes))

			each_class_id_size = mean_size // class_sz
			for i, class_id in enumerate(classes):
				selected_indices = data_indices[class_id][:each_class_id_size]
				data_indices[class_id] = data_indices[class_id][each_class_id_size:]
				participant_indices[participant_id].extend(selected_indices)

				# top up to make sure all participants have the same number of samples
				if i == len(classes) - 1 and len(participant_indices[participant_id]) < mean_size:
					extra_needed = mean_size - len(participant_indices[participant_id])
					participant_indices[participant_id].extend(data_indices[class_id][:extra_needed])
					data_indices[class_id] = data_indices[class_id][extra_needed:]
		train_indices = [participant_index_list for participant_id, participant_index_list in participant_indices.items()] 

	else:
		train_indices = np.array_split(indices, n_participants)
	return train_indices


def average_models(models, device=None):
	final_model = copy.deepcopy(models[0])
	if device:
		models = [model.to(device) for model in models]
		final_model = final_model.to(device)

	full_param_lists = [ list(model.parameters()) for model in models]
	averaged_parameters =[ torch.stack( [parameters[i] for parameters in full_param_lists] ).mean(dim=0)	for i in range(len(full_param_lists[0])) ]

	for param, avg_param in zip(final_model.parameters(), averaged_parameters):
		param.data = avg_param.data
	return final_model

def compute_grad_update(old_model, new_model, device=None):
	# maybe later to implement on selected layers/parameters
	if device:
		old_model, new_model = old_model.to(device), new_model.to(device)
	return [(new_param.data - old_param.data) for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]


def add_update_to_model(model, update, weight=1.0, device=None):
	if not update: return model
	if device:
		model = model.to(device)
		update = [param.to(device) for param in update]
			
	for param_model, param_update in zip(model.parameters(), update):
		param_model.data += weight * param_update.data
	return model


from itertools import repeat

def repeater(arr):
    for loader in repeat(arr):
        for item in arr:
            yield item


import copy
import numpy as np
from .kernels import rbf, dot



def MMD_hat_squared(X, Y, kernel=dot):
	n, m = len(X), len(Y)
	
	X = X.reshape(len(X), -1)
	Y = Y.reshape(len(Y), -1)
	pairwise_X = kernel(X, X)
	pairwise_Y = kernel(Y, Y)
	pairwise_XY = kernel(X, Y)
	mmd = 1.0/(n*(n-1)) * (pairwise_X.sum() - pairwise_X.trace())  + 1.0/(m*(m-1))* (pairwise_Y.sum()-pairwise_Y.trace())
	mmd -= 2.0/(n*m)* pairwise_XY.sum()
	return mmd


def MMD_update(MMD, X, Y, x_new, A_t, B_t, kernel=dot):
	n, m = len(X), len(Y)

	A_t = -2.0/(n+1)*A_t + 1.0/(n*(n+1)) * kernel([x_new], X)
	B_t = (-1.0/(n+1)*B_t - 2.0/(m*(n+1)) * kernel([x_new], Y))
	updated_mmd = A_t + B_t + MMD
	
	return updated_mmd, A_t, B_t


def calculate_quantiles(D, k=1000, kernel=dot):
	mmds = []
	D = copy.deepcopy(D).reshape(len(D), -1)

	for i in range(k):
		np.random.shuffle(D)
		X = copy.deepcopy(D[:len(D)//2])
		Y = copy.deepcopy(D[len(D)//2:])
		mmds.append(MMD_hat_squared(X, Y, kernel))
	return sorted(mmds)


def get_quantile(alpha, D):
	assert 0<alpha <1, "alpha must be in range (0,1) exclusive."
	# D = np.append(X, Y, axis = 0)
	quantiles = calculate_quantiles(D)

	return quantiles[int(alpha*len(quantiles))]


def union(D, R):
	if len(R) == 0:
		return D
	else:
		return np.concatenate((D, R), axis=0)
