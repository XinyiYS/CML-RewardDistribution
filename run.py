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
	def __init__(self, feature_extractor, gp_layer, num_dim=10, grid_bounds=(-10., 10.)):
		super(DKLModel, self).__init__()
		self.feature_extractor = feature_extractor
		self.gp_layer = gp_layer

		# self.grid_bounds = grid_bounds
		# self.num_dim = num_dim
	
	def forward(self, x1, x2):        
		features1 = self.get_vae_features(x1)
		features2 = self.get_vae_features(x2)   
		mmd_2, Kxx_, Kxy, Kyy_ = mmd(features1.reshape(len(x1), -1), features2.reshape(len(x2), -1), k=self.gp_layer.covar_module)
		t_stat = t_statistic(mmd_2, Kxx_, Kxy, Kyy_)
		return mmd_2, t_stat
		# return mmd_2, t_stat, self.get_vae_loss(x1) + self.get_vae_loss(x2)
		# return mmd_2, t_stat, self.gp_layer(self.process_features(features1)), self.gp_layer(self.process_features(features2))

	def get_vae_loss(self, x):
		x_recon, latent_mu, latent_logvar = self.feature_extractor(x)
		return vae_loss(x_recon, x, latent_mu, latent_logvar)

	def get_vae_features(self, x):
		x_mu, x_logvar = self.feature_extractor.encoder(x)
		return self.feature_extractor.latent_sample(x_mu, x_logvar)

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
	
	# def forward(self, x1, x2):        
	#     features1 = self.get_features(x1)
	#     features2 = self.get_features(x2)   
	#     mmd_2, Kxx_, Kxy, Kyy_ = mmd(features1.reshape(len(x1), -1), features2.reshape(len(x2), -1), k=self.gp_layer.covar_module)
	#     t_stat = t_statistic(mmd_2, Kxx_, Kxy, Kyy_)        
	#     return mmd_2, t_stat, self.gp_layer(features1), self.gp_layer(features2)

# need a kernel collectively defined by gpytorch and a DNN
def objective(args, model, optimizer, trial, joint_loader, train_loaders, test_loaders, ):

	with gpytorch.settings.num_likelihood_samples(8):
		for epoch in range(args['epochs']):
			joint_loader_iter = tqdm.notebook.tqdm(joint_loader, desc=f"(Epoch {epoch}) Minibatch")
			# loaders = [joint_loader_iter] + train_loaders
			loaders = train_loaders
			model.train()
			for data in zip(*loaders):
				# data is of length 5 [(data1, target1)... (data5, target5)]
				data = list(data)
				for i in range(len(data)):
					data[i][0], data[i][1] = data[i][0].cuda(), data[i][1].cuda()    
	  
				optimizer.zero_grad()
				
				mmd_losses = torch.zeros(len(data), device=data[0][0].device, requires_grad=False)
				
				for i in range(len(data)):
					for j in range(len(data)):
						mmd_2, t_stat = model(data[i][0], data[j][0])

						if torch.isnan(t_stat):
							print("t_stat is nan for {} vs {}, at {}-epoch".format(i, j, epoch+1))						
							obj = mmd_2
						else:
							obj = t_stat

						if i != j:
							mmd_losses[i] += obj
						else:
							mmd_losses[i] += -obj

						break # try to only optimize for the first participant

				# max min t_stats == min max -t_stats
				# pytorch optimization is minimization, so we take the max of -t_stat, and minimize it 
				
				# loss = torch.sum(torch.sign(mmd_losses) * torch.square(mmd_losses))
				loss = -torch.min(mmd_losses)
				# vae_loss = model.get_vae_loss(data_j) + model.get_vae_loss(data[i][0])
				# loss = torch.add(loss, vae_loss)
				loss.backward()
				optimizer.step()
				joint_loader_iter.set_postfix(loss=loss.item(), mmd_losses = mmd_losses.tolist())

			if epoch % args['save_interval'] == 0:
				torch.save(model.state_dict(), oj(args['logdir'], args['kernel_dir'], 'model_-E{}.pth'.format(epoch+1)))

			model.eval()
			mmd_pairwise = []
			with torch.no_grad():
				for i, loader in enumerate(test_loaders):
					mmds, _ = evaluate_pairwise(loader, test_loaders, model, M=50)
					mmd_pairwise.append(mmds)

			# --------------- Objective ---------------
			# small intra mmd, large inter mmd
			objective = 0 # to minimize
			for i in range(len(mmd_pairwise)):
				for j in range(len(mmd_pairwise)):
					if i==j:
						objective += mmd_pairwise[i][j].sum()
					else:
						objective -= mmd_pairwise[i][j].sum()

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
	vae = load_pretrain()
	feature_extractor = vae
	# feature_extractor = MLP_MNIST(in_dim = imageSize*imageSize, out_dim=num_features, device=device)

	# --------------- Gaussian Process/Kernel module ---------------

	grid_bounds=(-10., 10.)


	suggested_kernels = [getattr(gpytorch.kernels, base_kernel)(lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
					math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp)) for base_kernel in args['base_kernels'] ]
	covar_module = ScaleKernel(AdditiveKernel(*suggested_kernels))
	gp_layer = GaussianProcessLayer(covar_module=covar_module, num_dim=args['num_features'], grid_bounds=grid_bounds)

	# --------------- Complete Deep Kernel ---------------

	model = DKLModel(feature_extractor, gp_layer)

	if torch.cuda.is_available():
		if torch.cuda.device_count()>1:
			model = nn.DataParallel(model, device_ids = list(range(torch.cuda.device_count())))
		else:
			model = model.cuda()

	# ---------- Optimizer and Scheduler ----------
	optimizer = getattr(optim, args['optimizer'])([
		{'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
		{'params': model.gp_layer.hyperparameters(), 'lr': args['lr'] * 0.1},
		{'params': model.gp_layer.variational_parameters()},
	], lr=args['lr'], weight_decay=0)
	scheduler = MultiStepLR(optimizer, milestones=[0.5 * args['epochs'], 0.75 * args['epochs']], gamma=0.1)

	args['lr_scheduler'] = scheduler.__class__
	args['model'] = model.__class__
	args['feature_extractor'] = model.feature_extractor.__class__
	return model, optimizer

def evaluate(model, test_loaders, args):
	all_mmd_vs, all_tstats_vs = [], []
	model.eval()
	with torch.no_grad():
		for i, loader in enumerate(test_loaders):
			mmds, t_stats = evaluate_pairwise(loader, test_loaders, model, M=100)
			all_mmd_vs.append(mmds)
			all_tstats_vs.append(t_stats)
	all_mmd_vs = torch.stack(all_mmd_vs, dim=1)
	all_tstats_vs = torch.stack(all_tstats_vs, dim=1)

	mmd_dir = oj(args['logdir'], args['figs_dir'], 'mmd')
	tstat_dir = oj(args['logdir'], args['figs_dir'], 'tstat')
	os.makedirs(mmd_dir, exist_ok=True)
	os.makedirs(tstat_dir, exist_ok=True)

	for i in range(args['n_participants']):
		plot_mmd_vs(all_mmd_vs, index = i, save = oj(mmd_dir, '-'+str(i)), alpha=1e4)
		plot_mmd_vs(all_tstats_vs, index = i, save = oj(tstat_dir, '-'+str(i)), alpha=1, _type='tstat')
	return all_mmd_vs

def main(trial):
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
	evaluate(model, test_loaders, args)

	return obj_value




# from models.feature_extractors import CNN_MNIST, MLP_MNIST
from utils.utils import repeater, split, load_dataset, evaluate_pairwise, setup_experiment_dir, setup_dir, write_model, load_kernel
from utils.mmd import mmd, t_statistic
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

