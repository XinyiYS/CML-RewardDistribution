import os
from os.path import join as oj
from ast import literal_eval
from itertools import product
from collections import defaultdict
import json
import numpy as np
import math
import torch
import torchvision

from utils.mmd import mmd, t_statistic
from utils.utils import init_deterministic
from run import GaussianProcessLayer

def load_pickle_to_loaders(file_dir= 'CIFAR_cand_images.p'):
	synthetic = np.load(file_dir, allow_pickle=True)
	from torch.utils.data import DataLoader
	loaders = [DataLoader(data.astype(np.float32), batch_size=3000) for data in synthetic]
	return loaders

def pickle_features(kernel, loaders, name='CIFAR10_feat_custom'):
	feature_list = []
	candidate_loaders = load_pickle_to_loaders()
	with torch.no_grad():
		for loader in candidate_loaders:
			for i, data in enumerate(candidate_loaders[0]):
				data = data.cuda()
				data = data.permute(0, 3, 2, 1)
				features = kernel.extract_features(data)
				feature_list.append(features.cpu().numpy())
	import pickle
	with open(name + '.p', 'wb') as f:
		pickle.dump(feature_list, f)
	return


import gpytorch
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, AdditiveKernel

def contruct_kernel_only(split):

	if split == 'disjointclasses' or split == 'equal':
		ard_num_dims = 32
		base_kernel = 'MaternKernel'
	else:
		ard_num_dims = 16
		base_kernel = 'RBFKernel'
		
	grid_bounds=(-10., 10.)
	base_kernels = [getattr(gpytorch.kernels, base_kernel)(ard_num_dims=ard_num_dims,lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
					math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp))]
	covar_module = ScaleKernel(AdditiveKernel(*base_kernels))
	gp_layer = GaussianProcessLayer(covar_module=covar_module, num_dim=512, grid_bounds=grid_bounds)

	return gp_layer


if __name__=='__main__':

	init_deterministic(1234) # comment this out for faster runtime

	# --------- Initialize the kernel, including initializing and loading pretrained weights for the shared feature extrator 

	custom_split_kernel = contruct_kernel_only(split='custom').cuda()
	a = custom_split_kernel.load_state_dict(torch.load('CIFAR10_custom_kernel.pth'))
	print("Kernel weights loading status:", a)

	features = np.load('CIFAR10_feat_custom.p', allow_pickle=True)
	# convert to torch float tensor and put on GPU
	features = [torch.from_numpy(feature.astype(np.float32)).cuda() for feature in features]

	with torch.no_grad():
		'''
		Compute MMD, use T_stats as your algorithm

		'''
		mmd_hat, Kxx_, Kxy, Kyy_ = mmd(features[0], features[1], k=custom_split_kernel.covar_module)
		t_stat = t_statistic(mmd_hat, Kxx_, Kxy, Kyy_)

	equal_split_kernel = contruct_kernel_only(split='equal').cuda()
	a = equal_split_kernel.load_state_dict(torch.load('CIFAR10_equal_kernel.pth'))
	print("Kernel weights loading status:", a)

	features = np.load('CIFAR10_feat_equal.p', allow_pickle=True)
	# convert to torch float tensor and put on GPU
	features = [torch.from_numpy(feature.astype(np.float32)).cuda() for feature in features]

	with torch.no_grad():

		'''
		Compute MMD, use T_stats as your algorithm

		'''
		mmd_hat, Kxx_, Kxy, Kyy_ = mmd(features[0], features[1], k=equal_split_kernel.covar_module)
		t_stat = t_statistic(mmd_hat, Kxx_, Kxy, Kyy_)



