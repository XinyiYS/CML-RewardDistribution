import random
import numpy as np
import torch
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

from utils.utils import average_models, split

class Data_Prepper:
	def __init__(self, args):
		self.args_dict = args # as a helper variable for NLP datasets
		self.device = args['device']
				
		self.n_participants = args['n_participants']
		self.n_samples = args['n_samples']
		self.split_mode = args['split_mode']

		self.batch_size = args['batch_size']

		self.dataset = args['dataset']
		self.set_data_set(self.dataset)
		self.set_train_indices_list()
		self.test_size = args['test_size']
		self.set_loaders()


	def set_data_set(self, dataset):
		if dataset.lower() == 'mnist':
			self.train_dataset = FastMNIST(root='.data/mnist', train=True, download=True, device=self.device)
			self.test_dataset = FastMNIST(root='.data/mnist', train=False, download=True, device=self.device)

		if dataset.lower() == 'cifar10':
			self.train_dataset = FastCIFAR10('.data/cifar10', train=True, download=True, device=self.device)
			self.test_dataset = FastCIFAR10('.data/cifar10', train=False, download=True, device=self.device)

	def set_train_indices_list(self):
		self.train_indices_list = split(self.n_samples, self.n_participants, train_dataset=self.train_dataset, mode=self.split_mode)

	def set_loaders(self):

		self.train_loaders = [DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, sampler=SubsetRandomSampler(train_indices))  for train_indices in self.train_indices_list]

		test_indices = list(range(len(self.test_dataset)))
		random.shuffle(test_indices)
		sampler = SubsetRandomSampler(test_indices[:self.test_size])
		self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=10000, sampler=sampler)


def powerlaw(sample_indices, n_participants, alpha=1.65911332899, shuffle=False):
	# the smaller the alpha, the more extreme the division
	if shuffle:
		random.seed(1234)
		random.shuffle(sample_indices)

	from scipy.stats import powerlaw
	import math
	party_size = int(len(sample_indices) / n_participants)
	b = np.linspace(powerlaw.ppf(0.01, alpha), powerlaw.ppf(0.99, alpha), n_participants)
	shard_sizes = list(map(math.ceil, b/sum(b)*party_size*n_participants))
	indices_list = []
	accessed = 0
	for participant_id in range(n_participants):
		indices_list.append(sample_indices[accessed:accessed + shard_sizes[participant_id]])
		accessed += shard_sizes[participant_id]
	return indices_list



from torchvision.datasets import MNIST
class FastMNIST(MNIST):
	def __init__(self, *args, **kwargs):
		if 'device' in kwargs:
			device = kwargs['device']
			kwargs.pop('device')
		super().__init__(*args, **kwargs)		
		
		# self.data = self.data.float().div(255)
		self.data = self.data.unsqueeze(1).float().div(255)
		from torch.nn import ZeroPad2d
		# pad = ZeroPad2d(2)
		self.data = torch.stack([pad(sample.data) for sample in self.data])

		self.targets = self.targets.long()

		self.data = self.data.sub_(0.1307).div_(0.3081)
		# Put both data and targets on GPU in advance
		self.data, self.targets = self.data.to(device), self.targets.to(device)
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
		if 'device' in kwargs:
			device = kwargs['device']
			kwargs.pop('device')
		super().__init__(*args, **kwargs)
		
		# Scale data to [0,1]
		from torch import from_numpy
		self.data = from_numpy(self.data)
		self.data = self.data.float().div(255)
		self.data = self.data.permute(0, 3, 1, 2)

		self.targets = torch.Tensor(self.targets).long()

		# Normalize it with the usual CIFAR10 mean and std
		self.data = self.data.sub_(0.5).div_(0.5)
		# Put both data and targets on GPU in advance
		self.data, self.targets = self.data.to(device), self.targets.to(device)
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
