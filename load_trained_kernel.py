import os
from os.path import join as oj
from ast import literal_eval

import torch

from utils.utils import tabulate_dict, prepare_loaders, evaluate
from run import construct_kernel


if __name__=='__main__':

	# Experiment dir
	load_dir = 'server_logs/CML/logs/Experiment_2020-12-29-22-20/N1000-E70-B512'

	# Read the kernel architecture hyperparameters
	args = {} 
	with open(oj(load_dir,'settings_dict.txt') ,'r') as file:
		for line in file.readlines():
			(key, value) = line.strip().split(' : ', 1)

			try: 
				args[key] = literal_eval(value)
			except Exception as e:
				args[key] = value
	
	# Set the dataset parameters
	if 'dataset' not in args:
		args['dataset'] = 'MNIST'

	if 'include_joint' not in args:
		args['include_joint'] = False

	# Initialize the kernel
	kernel, _ = construct_kernel(args)

	# Load pretrained weights
	trained_kernel_dir = oj(load_dir, 'trained_kernels', 'model_-E61.pth')
	kernel.load_state_dict(torch.load(trained_kernel_dir), strict=False)


	# Construct data loaders for a quick evaluation
	joint_loader, train_loaders, joint_test_loader, test_loaders = prepare_loaders(args, repeat=args['include_joint'])

	test_logs_dir = "test_logs_dir"
	os.makedirs(test_logs_dir, exist_ok=True)
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