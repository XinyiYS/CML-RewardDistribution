import os
from os.path import join as oj
from ast import literal_eval

import torch

from utils.utils import tabulate_dict, prepare_loaders, evaluate, init_deterministic, load_kernel
from run import construct_kernel


if __name__=='__main__':

	# --------- Experiment dir
	load_dir = 'MNIST/N2000-E30-B832' # 'CIFAR10/N4000-E30-B640'

	# --------- Read the kernel architecture hyperparameters
	args = {} 
	with open(oj(load_dir,'settings_dict.txt') ,'r') as file:
		for line in file.readlines():
			(key, value) = line.strip().split(' : ', 1)

			try: 
				args[key] = literal_eval(value)
			except Exception as e:
				args[key] = value

	# --------- Can change the 'split_mode' and 'batch_size' at test time
	# args['split_mode'] = 'disjointclasses' #'uniform'
	# args['batch_size'] = 256

	init_deterministic(args['noise_seed']) # comment this out for faster runtime

	# --------- Initialize the kernel, including initializing and loading pretrained weights for the shared feature extrator 
	kernel, _ = construct_kernel(args)

	# --------- Load pretrained weights: including the individual MLP layers and the Gpytorch Hyperparameters
	kernel = load_kernel(kernel, oj(load_dir, 'trained_kernels'))


	# --------- Construct data loaders for a quick evaluation
	joint_loader, train_loaders, joint_test_loader, test_loaders = prepare_loaders(args, repeat=args['include_joint'])

	# --------- Create the directory to store test logs and figs
	test_logs_dir = oj("test_logs_dir"+'-'+args['split_mode'], args['dataset'])
	os.makedirs(test_logs_dir, exist_ok=True)
	if args['include_joint']:
		train_loaders = [joint_loader] + train_loaders
		test_loaders = [joint_test_loader] + test_loaders

	mmd_dict, tstat_dict = evaluate(kernel, test_loaders, args, M=50, plot=True, logdir=test_logs_dir, figs_dir = oj(test_logs_dir, 'figs'))

	mmd_mean_table, mmd_std_table = tabulate_dict(mmd_dict, args['n_participants'] + int(args['include_joint']))
	
	print("Pairwise MMD hat values mean: ")
	print(mmd_mean_table)
	print("Pairwise MMD hat values std: ")
	print(mmd_std_table)

	tstat_mean_table, tstat_std_table = tabulate_dict(tstat_dict, args['n_participants'] + int(args['include_joint']))
	print("Pairwise tStatitics mean: ")
	print(tstat_mean_table)
	print("Pairwise tStatitic std: ")
	print(tstat_std_table)