import math
import time, datetime
import os,sys
from os.path import join as oj

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch import optim
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

import tqdm
import gpytorch
from gpytorch.kernels import ScaleKernel, AdditiveKernel, RBFKernel, PolynomialKernel, MaternKernel


from run import GaussianProcessLayer, DKLModel, prepare_loaders, construct_kernel, evaluate
from utils.utils import repeater, split, load_dataset, evaluate_pairwise, setup_experiment_dir, setup_dir, write_model, load_kernel
from utils.mmd import mmd, t_statistic
from utils.plot import plot_mmd_vs

if __name__=='__main__':

	args = {}
	args['noise_seed'] = 1234

	# ---------- Data setting ----------
	n_participants = args['n_participants'] = 5
	args['n_samples_per_participant'] = 2000
	args['n_samples'] = args['n_participants'] * args['n_samples_per_participant']
	args['split_mode'] = "disjointclasses" #@param ["disjointclasses","uniform"," classimbalance", "powerlaw"]

	# ---------- Feature extractor and latent dim setting ----------

	args['num_features'] = 10 # latent_dims
	args['num_filters'] = 64 # fixed
	# ngf number of filters for encoder/generator
	# ndf number of filters for decoder/discriminator
	ngf = ndf = args['num_filters']
	# number of channels, 1 for MNIST, 3 for CIFAR-10, CIFAR-100
	args['num_channels'] = 1 
	args['num_classes'] = 10
	# fixed for MNIST
	args['image_size'] = 28

	# ---------- Optuna ----------
	args['epochs'] = None
	args['batch_size'] = None

	args['optimizer'] = None
	args['lr'] = None

	args['num_base_kernels'] = None
	args['base_kernels'] = None
	# ---------- Logging Directories ----------

	args['kernel_dir'] = 'trained_kernels'
	args['figs_dir'] = 'figs'
	args['save_interval'] = 10


	# Provide valid experiment_dir and logdir
	args['experiment_dir'] = 'logs/Experiment_2020-12-16-19-28'
	args['logdir'] = 'N2000-E60-B384'

	logdir = args['logdir'] = oj(args['experiment_dir'], args['logdir'])

	with open(oj(logdir,'settings_dict.txt'), 'r') as file:
		for line in file.readlines():
			[key, value] = line.strip().split(' : ', 1)
			if str.isdigit(value):
				args[key] = int(value)
			else:
				try:
					args[key] = float(value)
				except:
					args[key] = value

	args['num_base_kernels'] = 1
	args['base_kernels'] = ['RBFKernel']
	args['optimizer'] = 'SGD'

	model, optimizer = construct_kernel(args)
	joint_loader, repeated_train_loaders, test_loaders = prepare_loaders(args)

	kernel_dir = oj(args['logdir'], 'trained_kernels')
	model = load_kernel(model, kernel_dir)
	
	evaluate(model, test_loaders, args)