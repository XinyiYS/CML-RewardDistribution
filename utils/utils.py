import copy
import random
from collections import defaultdict

import numpy as np
import torch
from torch import nn

import matplotlib.pyplot as plt
import seaborn as sns
import json


def init_deterministic(seed=1234):
	# call init_deterministic() in each run_experiments function call

	torch.manual_seed(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	random.seed(seed)

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
		print("Using disjoint classes and partitioning the dataset of {} data to {} participants with each having {} classes.".format(len(train_dataset.data), n_participants, class_sz))

		clses = sorted([ cls for _, cls in zip(range(n_participants), repeater(cls_splits))])
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


def prepare_loaders(args, repeat=False):

	train_dataset, test_dataset = load_dataset(args=args)
	train_indices_list = split(args['n_samples'], args['n_participants'], train_dataset=train_dataset, mode=args['split_mode'])
	test_indices_list = split(len(test_dataset.data), args['n_participants'], train_dataset=test_dataset, mode=args['split_mode'])

	shuffle = True
	if shuffle:
		from random import shuffle
		for i in range(args['n_participants']):
			shuffle(train_indices_list[i])
			shuffle(test_indices_list[i])

	from torch.utils.data.sampler import SubsetRandomSampler
	from torch.utils.data import DataLoader

	train_loaders = [DataLoader(dataset=train_dataset, batch_size=args['batch_size'], sampler=SubsetRandomSampler(indices)) for indices in train_indices_list]
	test_loaders = [DataLoader(dataset=test_dataset, batch_size=args['batch_size'], sampler=SubsetRandomSampler(indices)) for indices in test_indices_list]

	import itertools
	train_indices = list(itertools.chain.from_iterable(train_indices_list))
	joint_loader = DataLoader(dataset=train_dataset, batch_size=args['batch_size'], sampler=SubsetRandomSampler(train_indices))
	# test_loader = DataLoader(dataset=test_dataset, batch_size=10000, shuffle=True)

	test_indices = list(itertools.chain.from_iterable(test_indices_list))
	joint_test_loader = DataLoader(dataset=test_dataset, batch_size=args['batch_size'], sampler=SubsetRandomSampler(test_indices))

	if not repeat:
		return joint_loader, train_loaders, joint_test_loader, test_loaders
	else:
		repeated_train_loaders = [repeater(train_loader) for train_loader in train_loaders]
		repeated_test_loaders = [repeater(test_loader) for test_loader in test_loaders]
		# repeated_joint_test_loader = repeater(joint_test_loader)
		# test_loaders = [repeated_joint_test_loader] + repeated_test_loaders
		return joint_loader, repeated_train_loaders, joint_test_loader, repeated_test_loaders




from itertools import repeat, product

def repeater(arr):
	for loader in repeat(arr):
		for item in arr:
			yield item


def tabulate_dict(pairwise_dict, N):
	from tabulate import tabulate

	pairs = list(product(range(N), range(N)))
	means = np.zeros((N, N))
	stds = np.zeros((N, N))
	for (i, j) in pairs:
		pair = str(i)+'-'+str(j)
		means[i, j] = np.mean(pairwise_dict[pair])
		stds[i, j] = np.std(pairwise_dict[pair])

	return tabulate(means, headers=range(N)), tabulate(stds, headers=range(N))



def evaluate(model, test_loaders, args, M=50, plot=False, logdir=None, figs_dir=None):
	'''
	Arguments:
		model: trained kernel, takes two inputs of the same shape
		test_loaders: a list of data loaders
		args: a dictionary used to store setting parameters
		M: number of samples to draw for the t-statistic
		plot: whether to plot the mmd and stats
		logdir: the directy to store the pairwise mmd and tstat values
		figs_dir: the directory to store the plotted figures
	Returns:
		mmd_dict, tstat_dict
		mmd_dict: 
			key is pairwise named, such as '1-1' participant 1 against himself, '0-3' participant 0 against participant 3
			value is a list of length M, representing M sampled pairs of values
		tstat_dict is defined similarly to mmd_dict
	'''

	N = args['n_participants'] + int(args['include_joint'])
	assert N == len(test_loaders), "The number of loaders is not equal equal to the total number of (paricipants + joint)."
	pairs = list(product(range(N), range(N)))
	model.eval()
	mmd_dict = defaultdict(list)
	tstat_dict = defaultdict(list)
	with torch.no_grad():
		for data in zip(*test_loaders):
			data = list(data)
			for i in range(len(data)):
				data[i][0], data[i][1] = data[i][0].cuda(), data[i][1].cuda()    					

			for m in range(M):
				for (i,j) in pairs:
					if i != j:
						size = len(data[i][0]) + len(data[j][0])
						temp = torch.cat([data[i][0], data[j][0]])
					else:
						size = len(data[i][0])
						temp = data[i][0]
					rand_inds =  torch.randperm(size)
					X, Y = temp[rand_inds[:size//2]], temp[rand_inds[size//2:]]
					mmd_hat, t_stat = model(X, Y, pair=[i, j])

					mmd_dict[str(i)+'-'+str(j)].append(mmd_hat.tolist())
					tstat_dict[str(i)+'-'+str(j)].append(t_stat.tolist())

	if not plot: 
		return mmd_dict, tstat_dict

	logdir = logdir or args['logdir']

	with open(oj(logdir, 'mmd_dict'), 'w') as file:
		file.write(json.dumps(mmd_dict))
	
	with open(oj(logdir, 'tstat_dict'), 'w') as file:
		file.write(json.dumps(tstat_dict))

	figs_dir = figs_dir or oj(args['logdir'], args['figs_dir'])
	mmd_dir = oj(figs_dir, 'mmd')
	tstat_dir = oj(figs_dir, 'tstat')
	os.makedirs(mmd_dir, exist_ok=True)
	os.makedirs(tstat_dir, exist_ok=True)


	# plot and save MMD hats	
	for i in range(N):
		for j in range(N):
			pair = str(i)+'-'+str(j)
			mmd_values = np.asarray(mmd_dict[pair])*1e4 
			sns.kdeplot(mmd_values, label=pair)

		plt.title('{} vs others MMD values'.format(str(i)))
		plt.xlabel('mmd values')
		# Set the y axis label of the current axis.
		plt.ylabel('density')
		# Set a title of the current axes.
		# show a legend on the plot
		plt.legend()
		# Display a figure.
		plt.savefig(oj(mmd_dir, '-'+str(i)))
		plt.clf()


	# plot and save Tstats	
	for i in range(N):
		for j in range(N):
			pair = str(i)+'-'+str(j)

			tstat_values = np.asarray(tstat_dict[pair])*1e3
			sns.kdeplot(tstat_values, label=pair)

		plt.title('{} vs others tstats values'.format(str(i)))
		plt.xlabel('tstat values')
		# Set the y axis label of the current axis.
		plt.ylabel('density')
		# Set a title of the current axes.
		# show a legend on the plot
		plt.legend()
		# Display a figure.
		plt.savefig(oj(tstat_dir, '-'+str(i)))
		plt.clf()

	return mmd_dict, tstat_dict



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
