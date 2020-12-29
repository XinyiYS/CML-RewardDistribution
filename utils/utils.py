import copy
import random
from collections import defaultdict

import numpy as np
import torch
from torch import nn

def split(n_samples, n_participants, train_dataset=None, mode='uniform'):
	'''
	Args:
	n_samples: total samples to be split among all participants
	n_participants: number of participants
	train_dataset: a torch dataset
	mode: split mode

	Returns:
	train_indices:[train_indices_1, train_indices2,...   ] a list of list of indices
	'''
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
		data_indices = [torch.nonzero(train_dataset.targets == class_id).view(-1).tolist() for class_id in range(n_classes)]

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
		data_indices = [torch.nonzero(train_dataset.targets == class_id).view(-1).tolist() for class_id in all_classes]

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


def evaluate_pairwise(self_loader, all_loaders, kernel, M=1000):
	n = len(all_loaders)
	mmds = torch.zeros((n, M))
	t_stats = torch.zeros((n, M))
	for i, other_loader in enumerate(all_loaders):
		mmd_hat = 0
		for m, (self_data, _), (other_data, _) in zip(range(M), self_loader, other_loader):
			self_data, other_data  = self_data.cuda(), other_data.cuda()
			mmd_2, t_stat, *_ = kernel(self_data, other_data)
			mmds[i][m] = mmd_2
			t_stats[i][m] = t_stat
	return mmds.detach(), t_stats.detach()


import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
def load_dataset(args):
	dataset = args['dataset']
	if dataset=='MNIST':        
		train_dataset = dset.MNIST(".data/mnist", train=True, download=True, 
			transform=transforms.Compose([transforms.Resize(args['image_size']), transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])]), )

		test_dataset = dset.MNIST(".data/mnist", train=False, download=True, 
			transform=transforms.Compose([transforms.Resize(args['image_size']), transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])]), )

	elif dataset == 'CIFAR10':
		
		train_dataset = FastCIFAR10('.data/cifar', train=True, download=True)
		test_dataset = FastCIFAR10('.data/cifar', train=False, download=True)
		print(train_dataset.data.shape)

		# transform_train = transforms.Compose([
		# 		transforms.RandomCrop(32, padding=4),
		# 		transforms.RandomHorizontalFlip(),
		# 		transforms.Resize(args['image_size']),
		# 		transforms.ToTensor(),
		# 		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		# 	])

		# transform_test = transforms.Compose([
		# 		transforms.ToTensor(),
		# 		transforms.Resize(args['image_size']),
		# 		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		# 	])

		# train_dataset = dset.CIFAR10('.data/cifar', train=True, download=True, transform=transform_train)
		# test_dataset = dset.CIFAR10('.data/cifar', train=False, download=True, transform=transform_test)
		# train_dataset.data = torch.from_numpy(train_dataset.data)
		# test_dataset.data = torch.from_numpy(test_dataset.data)
		# train_dataset.targets = torch.Tensor(train_dataset.targets).long()
		# test_dataset.targets = torch.Tensor(test_dataset.targets).long()

	return train_dataset, test_dataset


from torchvision.datasets import MNIST
class FastMNIST(MNIST):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)		
		
		self.data = self.data.unsqueeze(1).float().div(255)
		from torch.nn import ZeroPad2d
		pad = ZeroPad2d(2)
		self.data = torch.stack([pad(sample.data) for sample in self.data])

		self.targets = self.targets.long()

		self.data = self.data.sub_(self.data.mean()).div_(self.data.std())
		# self.data = self.data.sub_(0.1307).div_(0.3081)
		# Put both data and targets on GPU in advance
		if torch.cuda.is_available():
			self.data, self.targets = self.data.cuda(), self.targets.cuda()
		print('MNIST data shape {}, targets shape {}'.format(self.data.shape, self.targets.shape))

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, target = self.data[index], self.targets[index]

		return img, target

from torchvision.datasets import CIFAR10
class FastCIFAR10(CIFAR10):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		# Scale data to [0,1]
		from torch import from_numpy
		self.data = from_numpy(self.data)
		self.data = self.data.float().div(255)
		self.data = self.data.permute(0, 3, 1, 2)

		self.targets = torch.Tensor(self.targets).long()
		# https://github.com/kuangliu/pytorch-cifar/issues/16
		# https://github.com/kuangliu/pytorch-cifar/issues/8
		for i, (mean, std) in enumerate(zip((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))):
			self.data[:,i].sub_(mean).div_(std)

		# Put both data and targets on GPU in advance
		if torch.cuda.is_available():
			self.data, self.targets = self.data.cuda(), self.targets.cuda()
		print('CIFAR10 data shape {}, targets shape {}'.format(self.data.shape, self.targets.shape))

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, target = self.data[index], self.targets[index]

		return img, target

# --------------- Book-keeping helper functions ---------------

import time, datetime

import os, sys
from os.path import join as oj

def setup_experiment_dir(experiment_dir=None):

	ts = time.time()
	st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M')

	if experiment_dir:
		experiment_dir = os.path.join(experiment_dir, "Experiment_{}".format(st))
	else:
		experiment_dir = os.path.join("logs", "Experiment_{}".format(st))

	try:
		os.makedirs(experiment_dir, exist_ok=True)
	except Exception as e:
		print(str(e))
		pass
	print("Experimental directory is set up at: ", experiment_dir)
	return experiment_dir

def setup_dir(experiment_dir, args):
	subdir = "N{}-E{}-B{}".format(args['n_samples_per_participant'], 
							args['epochs'], args['batch_size'], 
							)
	logdir = os.path.join(experiment_dir, subdir)
	from sys import platform
	if platform in ['win32', 'cygwin']:
		# Windows
		logdir = os.path.join(os.getcwd(), logdir)
	elif platform == "linux" or platform == "linux2":
		# linux
		pass
	elif platform == "darwin":
		# OS X
		pass

	try:
		os.makedirs(logdir, exist_ok=True)
	except Exception as e:
		print("RAISING AN EXCEPTION:", str(e), "and logdir is:", logdir)
		pass

	if 'complete.txt' in os.listdir(logdir):
		return logdir

	with open(os.path.join(logdir,'settings_dict.txt'), 'w') as file:
		[file.write(key + ' : ' + str(value) + '\n') for key, value in args.items()]
	return logdir

def write_model(model, logdir, args):
	original_stdout = sys.stdout # Save a reference to the original standard output
	from torchsummary import summary
	with open(oj(logdir, 'model_summary.txt'), 'w') as file:
		sys.stdout = file # Change the standard output to the file we created.
		print(model.named_modules)
		file.write('\n')
		file.write('\n')
		file.write('\n')
		file.write('\n ----------------------- Feature Extractor Summary ----------------------- \n')
		summary(model.feature_extractor, input_size=(args['num_channels'], args['image_size'], args['image_size']), batch_size=args['batch_size'])
		sys.stdout = original_stdout # Reset the standard output to its original value
	return

def load_kernel(model, kernel_dir='trained_kernels', latest=True):
	max_E = max([int(kernel[kernel.find('E')+1: kernel.find('.pth')]) for kernel in os.listdir(kernel_dir)])
	model.load_state_dict(torch.load(oj(kernel_dir, 'model_-E{}.pth'.format(str(max_E)))))
	return model
