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
import gpytorch
from gpytorch.kernels import ScaleKernel, AdditiveKernel, RBFKernel, PolynomialKernel, MaternKernel

class GaussianProcessLayer(gpytorch.models.ApproximateGP):
	def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64, covar_module=None):
		variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
			num_inducing_points=grid_size, batch_shape=torch.Size([num_dim]))
		
		# Our base variational strategy is a GridInterpolationVariationalStrategy,
		# which places variational inducing points on a Grid
		# We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
		variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
			gpytorch.variational.GridInterpolationVariationalStrategy(
				self, grid_size=grid_size, grid_bounds=[grid_bounds],
				variational_distribution=variational_distribution,
			), num_tasks=num_dim,
		)
		super().__init__(variational_strategy)
		
		if covar_module:
			self.covar_module = covar_module
		else:
			self.covar_module = gpytorch.kernels.ScaleKernel(
				gpytorch.kernels.RBFKernel(
					active_dims=torch.tensor([0,1,2,3,4]),
					lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
						math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
					)
				) + gpytorch.kernels.RBFKernel(
					active_dims=torch.tensor([5,6,7,8,9]),
					lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
						math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
					)
				) 
			) 

		self.mean_module = gpytorch.means.ConstantMean()
		self.grid_bounds = grid_bounds

	def forward(self, x):
		mean = self.mean_module(x)
		covar = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean, covar)

class DKLModel(gpytorch.Module):
	def __init__(self, feature_extractor, indi_feature_extractors, gp_layer, num_dim=10, grid_bounds=(-10., 10.)):
		super(DKLModel, self).__init__()
		self.feature_extractor = feature_extractor
		self.indi_feature_extractors = indi_feature_extractors
		self.gp_layer = gp_layer

	
	def forward(self, x1, x2, pair, A=None, B=None, C=None):
		features1 = self.get_vae_features(x1)
		features2 = self.get_vae_features(x2)

		features1 = self.indi_feature_extractors[pair[0]](features1)
		features2 = self.indi_feature_extractors[pair[1]](features2)
		mmd_2, Kxx_, Kxy, Kyy_ = mmd(features1.reshape(len(x1), -1), features2.reshape(len(x2), -1), k=self.gp_layer.covar_module)
		t_stat = t_statistic(mmd_2, Kxx_, Kxy, Kyy_)
		# if A is None:
		# else:
			# assert A is not None and B is not None and C is not None, "Batch update missing previous pairwise estimates"
			# mmd_update_batch(x, features1, features2, A, B, C, k)
		# t_stat = t_statistic(mmd_2, Kxx_, Kxy, Kyy_)
		
		return mmd_2, t_stat
		# return mmd_2, t_stat, self.get_vae_loss(x1) + self.get_vae_loss(x2)
		# return mmd_2, t_stat, self.gp_layer(self.process_features(features1)), self.gp_layer(self.process_features(features2))
	
	def get_vae_features(self, x):
		x_mu, x_logvar = self.feature_extractor.encoder(x)
		return self.feature_extractor.latent_sample(x_mu, x_logvar)

	'''
	def get_vae_loss(self, x):
		x_recon, latent_mu, latent_logvar = self.feature_extractor(x)
		return vae_loss(x_recon, x, latent_mu, latent_logvar)


	def get_features(self, x):
		features = self.feature_extractor(x)
		features = gpytorch.utils.grid.scale_to_bounds(features, self.grid_bounds[0], self.grid_bounds[1])
		# This next line makes it so that we learn a GP for each feature
		features = features.transpose(-1, -2).unsqueeze(-1)
		return features
	
	def process_features(self, features):
		features = gpytorch.utils.grid.scale_to_bounds(features, self.grid_bounds[0], self.grid_bounds[1])
		# This next line makes it so that we learn a GP for each feature
		features = features.transpose(-1, -2).unsqueeze(-1)
		return  features
	
	def forward(self, x1, x2):        
		features1 = self.get_features(x1)
		features2 = self.get_features(x2)   
		mmd_2, Kxx_, Kxy, Kyy_ = mmd(features1.reshape(len(x1), -1), features2.reshape(len(x2), -1), k=self.gp_layer.covar_module)
		t_stat = t_statistic(mmd_2, Kxx_, Kxy, Kyy_)        
		return mmd_2, t_stat, self.gp_layer(features1), self.gp_layer(features2)
	'''
# A simple individual feature extrator on top of the shared feature extrators
class MLP(nn.Module):
	def __init__(self, input_dim=10, output_dim=5, device=None):
		super(MLP, self).__init__()
		self.fc1 = nn.Linear(input_dim, 10)
		self.fc2 = nn.Linear(10, output_dim)

	def forward(self, x):
		x = torch.sigmoid(self.fc1(x))
		return self.fc2(x)


# need a kernel collectively defined by gpytorch and a DNN
def objective(args, model, optimizer, trial, joint_loader, train_loaders, test_loaders, ):

	N = args['n_participants']
	pairs = list(product(range(N), range(N)))
	with gpytorch.settings.num_likelihood_samples(8):
		for epoch in range(args['epochs']):
			joint_loader_iter = tqdm.notebook.tqdm(joint_loader, desc=f"(Epoch {epoch}) Minibatch")
			# loaders = [joint_loader_iter] + train_loaders
			loaders = train_loaders
			model.train()

			for batch_id, data in enumerate(zip(*loaders)):
				# data is of length 5 [(data1, target1)... (data5, target5)]
				data = list(data)
				for i in range(len(data)):
					data[i][0], data[i][1] = data[i][0].cuda(), data[i][1].cuda()    
	  
				optimizer.zero_grad()
			
				mmd_losses = torch.zeros(N, device=data[0][0].device, requires_grad=False)
				for (i, j) in pairs:
					mmd_2, t_stat = model(data[i][0], data[j][0], pair=[i,j])
					if torch.isnan(t_stat):
						print("t_stat is nan for {} vs {}, at {}-epoch".format(i, j, epoch+1))						
						obj = mmd_2
					else:
						obj = t_stat

					if i != j:
						mmd_losses[i] += obj
					else:
						mmd_losses[i] += -obj

				loss = -torch.min(mmd_losses)
				loss.backward()
				optimizer.step()
				joint_loader_iter.set_postfix(loss=loss.item(), mmd_losses = mmd_losses.tolist())

			if epoch % args['save_interval'] == 0:
				torch.save(model.state_dict(), oj(args['logdir'], args['kernel_dir'], 'model_-E{}.pth'.format(epoch+1)))

			mmd_dict, tstat_dict = evaluate(model, test_loaders, args, M=50, plot=False)
			# --------------- Objective ---------------
			# small intra mmd, large inter mmd
			objective = 0 # to minimize
			for (i,j) in pairs:
				if i==j:
					objective += sum(mmd_dict[str(i)+'-'+str(j)])
				else:
					objective -= sum(mmd_dict[str(i)+'-'+str(j)])

			trial.report(objective, epoch)
			# Handle pruning based on the intermediate value.
			if trial.should_prune():
				raise optuna.exceptions.TrialPruned()
	return objective

def prepare_loaders(args):

	train_dataset, test_dataset = load_dataset(args=args)
	args['n_participants']
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
	test_loaders = [DataLoader(dataset=train_dataset, batch_size=args['batch_size'], sampler=SubsetRandomSampler(indices)) for indices in test_indices_list]

	import itertools
	train_indices = list(itertools.chain.from_iterable(train_indices_list))
	joint_loader = DataLoader(dataset=train_dataset, batch_size=args['batch_size'], sampler=SubsetRandomSampler(train_indices))
	# test_loader = DataLoader(dataset=test_dataset, batch_size=10000, shuffle=True)

	test_indices = list(itertools.chain.from_iterable(test_indices_list))
	joint_test_loader = DataLoader(dataset=test_dataset, batch_size=args['batch_size'], sampler=SubsetRandomSampler(test_indices))

	# repeated_train_loaders = [repeater(train_loader) for train_loader in train_loaders]
	# repeated_test_loaders = [repeater(test_loader) for test_loader in test_loaders]

	# repeated_joint_test_loader = repeater(joint_test_loader)
	# test_loaders = [repeated_joint_test_loader] + repeated_test_loaders

	return joint_loader, train_loaders, test_loaders

	# return joint_loader, repeated_train_loaders, test_loaders

def construct_kernel(args):

	from models.CVAE import VariationalAutoencoder, load_pretrain

	# --------------- Feature extractor module ---------------


	# --------------- Shared Feature extractor module ---------------

	vae = load_pretrain()
	feature_extractor = vae
	# feature_extractor = MLP_MNIST(in_dim = imageSize*imageSize, out_dim=num_features, device=device)

	# --------------- Individual layers after the Shared Feature extractor module ---------------
	indi_feature_extractors = [MLP(input_dim=args['num_features'], output_dim=args['num_features']) for _ in range(args['n_participants'])]

	# --------------- Gaussian Process/Kernel module ---------------
	grid_bounds=(-10., 10.)
	suggested_kernels = [getattr(gpytorch.kernels, base_kernel)(lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
					math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp)) for base_kernel in args['base_kernels'] ]
	covar_module = ScaleKernel(AdditiveKernel(*suggested_kernels))
	gp_layer = GaussianProcessLayer(covar_module=covar_module, num_dim=args['num_features'], grid_bounds=grid_bounds)

	# --------------- Complete Deep Kernel ---------------


	model = DKLModel(feature_extractor, indi_feature_extractors, gp_layer)

	if torch.cuda.is_available():
		if torch.cuda.device_count()>1:
			model = nn.DataParallel(model, device_ids = list(range(torch.cuda.device_count())))
		else:
			model = model.cuda()
			model.feature_extractor = model.feature_extractor.cuda()
			model.indi_feature_extractors = [f.cuda() for f in model.indi_feature_extractors]

	# ---------- Optimizer and Scheduler ----------
	optimizer = getattr(optim, args['optimizer'])([
		{'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
		{'params': model.gp_layer.hyperparameters(), 'lr': args['lr'] * 0.1, 'weight_decay':1e-4},
		{'params': model.gp_layer.variational_parameters(), 'weight_decay':1e-4},
	], lr=args['lr'], weight_decay=0)
	scheduler = MultiStepLR(optimizer, milestones=[0.5 * args['epochs'], 0.75 * args['epochs']], gamma=0.1)

	args['lr_scheduler'] = scheduler.__class__
	args['model'] = model.__class__
	args['feature_extractor'] = model.feature_extractor.__class__
	return model, optimizer

def evaluate(model, test_loaders, args, M=50, plot=False):
	N = args['n_participants']
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

					mmd_dict[str(i)+'-'+str(j)].append(mmd_hat)
					tstat_dict[str(i)+'-'+str(j)].append(t_stat)

	if not plot: 
		return mmd_dict, tstat_dict
	
	mmd_dir = oj(args['logdir'], args['figs_dir'], 'mmd')
	tstat_dir = oj(args['logdir'], args['figs_dir'], 'tstat')
	os.makedirs(mmd_dir, exist_ok=True)
	os.makedirs(tstat_dir, exist_ok=True)

	for i in range(args['n_participants']):
		for key, value in mmd_dict.items():
			if str(i) in key:
				value = np.asarray(value)*10
				sns.kdeplot(value, label=key)

				plt.title('{} vs others MMD values'.format(str(i+1)))
				plt.xlabel('mmd values')
				# Set the y axis label of the current axis.
				plt.ylabel('density')
				# Set a title of the current axes.
				# show a legend on the plot
				plt.legend()
				# Display a figure.
				plt.savefig(oj(mmd_dir, '-'+str(i)))
				plt.clf()


				value = np.asarray(tstat_dict[key])*10
				sns.kdeplot(value, label=key)

				plt.title('{} vs others tstats values'.format(str(i+1)))
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

def main(trial):
	args = {}
	args['noise_seed'] = 1234

	# ---------- Data setting ----------

	n_participants = args['n_participants'] = 5
	args['n_samples_per_participant'] = 1000
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
	args['epochs'] = trial.suggest_int("epochs", 10, 100, 5)
	args['batch_size'] = trial.suggest_int("batch_size", 512, 1024, 128)

	args['optimizer'] = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
	args['lr'] = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

	args['num_base_kernels'] = trial.suggest_int("num_base_kernels", 1, 3, 1)
	args['base_kernels'] = [trial.suggest_categorical('kernel{}_name'.format(i+1), ['RBFKernel', 'MaternKernel']) for i in range(args['num_base_kernels'])]

	# ---------- Logging Directories ----------

	args['kernel_dir'] = 'trained_kernels'
	args['figs_dir'] = 'figs'
	args['save_interval'] = 10

	args['train'] = True # if False, load the model from <experiment_dir> for evaluation
	args['experiment_dir'] = 'logs/Experiment_2020-12-16-19-28'

	# --------------- Create and Load Model ---------------

	model, optimizer = construct_kernel(args)

	# --------------- Set up the experiment config with selected hyperparameters---------------

	args['experiment_dir'] = setup_experiment_dir()
	logdir = setup_dir(args['experiment_dir'] , args)
	args['logdir'] = logdir

	write_model(model, logdir, args)	
	sys.stdout = open(os.path.join(logdir, 'log'), "w")

	# --------------- Preparing Datasets and Dataloaders ---------------
	joint_loader, train_loaders, test_loaders = prepare_loaders(args)

	# --------------- Training ---------------
	os.makedirs(oj(args['logdir'], args['kernel_dir']) , exist_ok=True)

	obj_value = objective(args, model, optimizer, trial, joint_loader, train_loaders, test_loaders)

	# --------------- Evaluating Performance ---------------
	evaluate(model, test_loaders, args, M=100, plot=True)

	return obj_value




# from models.feature_extractors import CNN_MNIST, MLP_MNIST
from utils.utils import repeater, split, load_dataset, evaluate_pairwise, setup_experiment_dir, setup_dir, write_model, load_kernel
from utils.mmd import mmd, mmd_update_batch, t_statistic
from utils.plot import plot_mmd_vs

import optuna
if __name__=='__main__':
	study = optuna.create_study(direction="minimize")
	study.optimize(main, n_trials=100)

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

