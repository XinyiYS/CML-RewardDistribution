import math
import time, datetime
import os,sys
from os.path import join as oj
from collections import defaultdict
from itertools import product
import numpy as np
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

from utils.mmd import Pdist2, h1_mean_var_gram
class DKL(nn.Module):

	def __init__(self, feature_extractor, **kwargs):

		super(DKL, self).__init__(**kwargs)

		self.feature_extractor = feature_extractor

		self.epsilon = torch.nn.Parameter(torch.tensor(1e-1))
		self.sigma_K = torch.nn.Parameter(torch.tensor(1e-1))
		self.sigma_q = torch.nn.Parameter(torch.tensor(1e-1))

	def forward(self, X, Y):
		L = 1 # generalized Gaussian (if L>1)

		nx = X.shape[0]
		ny = Y.shape[0]

		featuresX, featuresY = self.feature_extractor(Y), self.feature_extractor(Y)

		Dxx = Pdist2(featuresX, featuresX)
		Dyy = Pdist2(featuresX, featuresY)
		Dxy = Pdist2(featuresX, featuresY)

		X = X.view(X.shape[0], -1)
		Y = Y.view(Y.shape[0], -1)
		Dxx_org = Pdist2(X, X)
		Dyy_org = Pdist2(Y, Y)
		Dxy_org = Pdist2(X, Y)
		K_Ix = torch.eye(nx).cuda()
		K_Iy = torch.eye(ny).cuda()

		Kx = (1-self.epsilon) * torch.exp(-(Dxx / self.sigma_K)**L -Dxx_org / self.sigma_q) + self.epsilon * torch.exp(-Dxx_org / self.sigma_q)
		Ky = (1-self.epsilon) * torch.exp(-(Dyy / self.sigma_K)**L -Dyy_org / self.sigma_q) + self.epsilon * torch.exp(-Dyy_org / self.sigma_q)
		Kxy = (1-self.epsilon) * torch.exp(-(Dxy / self.sigma_K)**L -Dxy_org / self.sigma_q) + self.epsilon * torch.exp(-Dxy_org / self.sigma_q)

		mmd2, varEst, Kxyxy = h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=True, use_1sample_U=True)

		mmd_hat = -1 * mmd2
		mmd_std_hat = torch.sqrt(varEst + 10 ** (-8))

		return mmd_hat, mmd_std_hat, torch.div(mmd_hat, mmd_std_hat)
		 
def objective(args, model, optimizer, trial, train_loaders, test_loaders):

	N = args['n_participants'] + int(args['include_joint'])
	pairs = list(product(range(N), range(N)))
	torch.save(model.state_dict(), oj(args['logdir'], args['kernel_dir'], 'model_-E{}.pth'.format(0)))
	for epoch in range(args['epochs']):
		model.train()
		for batch_id, data in enumerate(zip(*train_loaders)):
			# data is of length 5 [(data1, target1)... (data5, target5)]
			data = list(data)
			for i in range(len(data)):
				data[i][0], data[i][1] = data[i][0].cuda(), data[i][1].cuda()    
  
			optimizer.zero_grad()
		
			mmd_losses = torch.zeros(N, device=data[0][0].device, requires_grad=False)
			for (i, j) in pairs:

				if i != j:
					X, Y = data[i][0], data[j][0]
				else:
					size = len(data[i][0])
					temp = data[i][0]
					rand_inds =  torch.randperm(size)
					X, Y = temp[rand_inds[:size//2]], temp[rand_inds[size//2:]]

				if X.size(0) < 4 or Y.size(0) < 4: continue
				# Too small a batch leftover, would cause the t-statistic to be undefined, so skip
				if X.size(0) != Y.size(0): continue

				mmd_hat, mmd_std_hat, t_stat = model(X, Y)
				print(t_stat.shape)
				if torch.isnan(t_stat):
					print("t_stat is nan for {} vs {}, at {}-epoch".format(i, j, epoch+1))						
					obj = mmd_hat
				else:
					obj = t_stat

				if i != j:
					mmd_losses[i] += obj
				else:
					mmd_losses[i] += -obj

			loss = -torch.min(mmd_losses)
			loss.backward()
			optimizer.step()

		if (epoch+1) % args['save_interval'] == 0:
			torch.save(model.state_dict(), oj(args['logdir'], args['kernel_dir'], 'model_-E{}.pth'.format(epoch+1)))

		mmd_dict, tstat_dict = evaluate(model, test_loaders, args, plot=False)
		
		# --------------- Objective ---------------
		# small intra mmd, large inter mmd
		obj = 0 # to minimize
		for (i,j) in pairs:
			if i==j:
				obj += sum(mmd_dict[str(i)+'-'+str(j)])
			else:
				obj -= sum(mmd_dict[str(i)+'-'+str(j)])

		trial.report(obj, epoch)
		# Handle pruning based on the intermediate value.
		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()

	torch.save(model.state_dict(), oj(args['logdir'], args['kernel_dir'], 'model_-E{}.pth'.format(args['epochs'])))
	return obj


def construct_kernel(args):

	# --------------- Feature extractor module ---------------
	MLP_feature_extractor = None

	# --------------- Shared Feature extractor module ---------------
	if args['dataset'] == 'CIFAR10':
		# from models.CIFAR_CVAE import CIFAR_CVAE, load_pretrain
		# CVAE = CIFAR_CVAE(latent_dims=args['num_features'])
		# feature_extractor = load_pretrain(vae=CVAE, path='CIFAR10_CVAE/model_512d.pth') # latent dimension is 512
	
		from models.CIFAR_Featurizer import Featurizer
		feature_extractor = Featurizer()

	else:
		# MNIST
		# from models.CVAE import VariationalAutoencoder, load_pretrain
		# vae = load_pretrain()
		# feature_extractor = vae

		# --------------- Individual layers after the Shared Feature extractor module ---------------

		# # MLP feature extractor on top of the VAEs		
		from models.feature_extractors import MLP, MLP_MNIST
		feature_extractor = MLP_MNIST()


	# --------------- Complete Deep Kernel ---------------
	model = DKL(feature_extractor)
	
	# ---------- Optimizer and Scheduler ----------
	optimizer = getattr(optim, args['optimizer'])([
				{'params': model.feature_extractor.parameters(), 'lr': args['lr'], 'weight_decay': 1e-4},
				{'params': model.epsilon, 'lr': args['lr']* 1e-2},
				{'params': model.sigma_K, 'lr': args['lr']* 1e-2},
				{'params': model.sigma_q, 'lr': args['lr']* 1e-2},
			], lr=args['lr'], weight_decay=0)
	scheduler = MultiStepLR(optimizer, milestones=[0.5 * args['epochs'], 0.75 * args['epochs']], gamma=0.95)


	if 'DataParallel' in model.__repr__():
		args['model'] = model.module.__class__
		args['feature_extractor'] = model.module.feature_extractor.__class__
	else:
		args['model'] = model.__class__
		args['feature_extractor'] = model.feature_extractor.__class__

	args['lr_scheduler'] = scheduler.__class__
	return model, optimizer

def train_main(trial):
	args = {}
	args['noise_seed'] = 1234

	init_deterministic(args['noise_seed']) # comment this out for faster training

	# ---------- Data setting ----------
	args['dataset'] = 'CIFAR10' # CIFAR10, MNIST


	args['split_mode'] = "custom" #@param ["disjointclasses","uniform"," classimbalance", "powerlaw", 'custom']
	args['clses'] = [[0],[1],[6],[8],[9]] if args['dataset'] == 'CIFAR10' else None
	args['clses'] = None

	args['include_joint'] = True

	n_participants = args['n_participants'] = 5
	args['n_samples_per_participant'] = 2000 
	args['n_samples_per_participant_test'] = 1000
	# args['class_sz_per_participant'] = 2
	args['n_samples'] = args['n_participants'] * args['n_samples_per_participant']
	args['n_samples_test'] = args['n_participants'] * args['n_samples_per_participant_test']
	

	# ---------- Feature extractor and latent dim setting ----------

	args['num_features'] = 512 if args['dataset'] == 'CIFAR10' else 10 # latent_dims
	args['num_filters'] = 64 # fixed
	# ngf number of filters for encoder/generator
	# ndf number of filters for decoder/discriminator
	ngf = ndf = args['num_filters']
	
	# number of channels, 1 for MNIST, 3 for CIFAR-10, CIFAR-100	
	args['num_channels'] = 3 if args['dataset'] == 'CIFAR10' else 1
	args['image_size'] = 32 if args['dataset'] == 'CIFAR10' else 28

	# learning the lengscale hyperparameters individually for each dimension
	args['ard_num_dims'] = True if args['dataset'] == 'CIFAR10' else False

	# fixed for MNIST and CIFAR10
	args['num_classes'] = 10

	# ---------- Optuna ----------
	args['epochs'] = trial.suggest_int("epochs", 25, 200, 25)
	# args['epochs'] = 0
	args['batch_size'] = trial.suggest_int("batch_size", 256, 1024, 64)
	# args['batch_size'] = 4

	args['optimizer'] = "Adam"
	args['lr'] = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

	args['num_base_kernels'] = 1
	args['base_kernels'] = ['RBFKernel']

	# ---------- Logging Directories ----------

	args['kernel_dir'] = 'trained_kernels'
	args['figs_dir'] = 'figs'
	args['save_interval'] = 25

	# --------------- Create and Load Model ---------------

	model, optimizer = construct_kernel(args)

	# --------------- Set up the experiment config with selected hyperparameters---------------

	args['experiment_dir'] = setup_experiment_dir()
	logdir = setup_dir(args['experiment_dir'] , args)
	args['logdir'] = logdir
	
	write_model(model, logdir, args)	
	# sys.stdout = open(os.path.join(logdir, 'log'), "w")


	# --------------- Make use of multiple GPUs if available ---------------
	if torch.cuda.is_available():
		model = model.cuda()
		if torch.cuda.device_count() > 0:
			model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))


	# --------------- Preparing Datasets and Dataloaders ---------------
	joint_loader, train_loaders, joint_test_loader, test_loaders = prepare_loaders(args, repeat=args['include_joint'])

	# --------------- Training ---------------
	os.makedirs(oj(args['logdir'], args['kernel_dir']) , exist_ok=True)

	if args['include_joint']:
		train_loaders = [joint_loader] + train_loaders
		test_loaders = [joint_test_loader] + test_loaders

	obj_value = objective(args, model, optimizer, trial, train_loaders, test_loaders)

	# --------------- Evaluating Performance ---------------
	mmd_dict, tstat_dict = evaluate(model, test_loaders, args, plot=True)

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

	return obj_value


def run():
	study = optuna.create_study(direction="minimize")
	study.optimize(train_main, n_trials=100)

	pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
	complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

	print("Study statistics: ")
	print("  Number of finished trials: ", len(study.trials))
	print("  Number of pruned trials: ", len(pruned_trials))
	print("  Number of complete trials: ", len(complete_trials))

	print("Best trial:")
	trial = study.best_trial

	print("  Value: ", trial.value)

	print("  Params: ")
	for key, value in trial.params.items():
		print("    {}: {}".format(key, value))


# from models.feature_extractors import CNN_MNIST, MLP_MNIST
from utils.utils import evaluate, prepare_loaders, init_deterministic, setup_experiment_dir, setup_dir, write_model, tabulate_dict
from utils.mmd import mmd, t_statistic


import optuna
if __name__=='__main__':
	run()

