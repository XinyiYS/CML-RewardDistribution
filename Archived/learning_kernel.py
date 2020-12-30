"""
partition the data into 5, plus 1 joint dataset

train a kernel end-to-end based on mini-batched of MMD estimate



"""

# from utils.Data_Prepper import Data_Prepper

# data_prepper = Data_Prepper()
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda')


import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

train_dataset = dset.MNIST(".data/mnist", train=True, download=True, 
    transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])]), )

test_dataset = dset.MNIST(".data/mnist", train=False, download=True, 
    transform=transforms.Compose([transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.1307], [0.3081])]), )

n_samples = 600
n_participants = 5
split_mode = 'disjointclasses'

from utils.utils import split
train_indices_list = split(n_samples, n_participants, train_dataset=train_dataset, mode=split_mode)

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

batch_size = 32

train_loaders = [DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))  for train_indices in train_indices_list]
joint_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=10000, shuffle=True)

import gpytorch



class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )
        
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
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
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

# for MNIST 28*28
class MLP_MNIST(nn.Module):
	def __init__(self, device=None):
		super(MLP_MNIST, self).__init__()
		self.fc1 = nn.Linear(784, 512)
		self.fc2 = nn.Linear(512, 128)
		self.fc3 = nn.Linear(128, 32)
		self.fc4 = nn.Linear(32, 2)

	def forward(self, x):
		x = x.view(-1,  784)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return x
		# return F.log_softmax(x, dim=1)

feature_extractor = MLP_MNIST(device=device)
num_features = 2
num_classes = 10
class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, num_dim, grid_bounds=(-10., 10.)):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

    def forward(self, x):
        features = self.feature_extractor(x)
        features = gpytorch.utils.grid.scale_to_bounds(features, self.grid_bounds[0], self.grid_bounds[1])
        # This next line makes it so that we learn a GP for each feature
        features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)
        return res

model = DKLModel(feature_extractor, num_dim=num_features)
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=model.num_dim, num_classes=num_classes)

# If you run this example without CUDA, I hope you like waiting!
if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()


from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

n_epochs = 1
lr = 0.1
optimizer = SGD([
    {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
    {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
    {'params': model.gp_layer.variational_parameters()},
    {'params': likelihood.parameters()},
], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)
mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(joint_loader.dataset))


def train(epoch):
    model.train()
    likelihood.train()

    minibatch_iter = tqdm.notebook.tqdm(joint_loader, desc=f"(Epoch {epoch}) Minibatch")
    with gpytorch.settings.num_likelihood_samples(8):
        for data, target in minibatch_iter:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = -mll(output, target)
            loss.backward()
            optimizer.step()
            minibatch_iter.set_postfix(loss=loss.item())
        
def test():
    model.eval()
    likelihood.eval()

    correct = 0
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = likelihood(model(data))  # This gives us 16 samples from the predictive distribution
            pred = output.probs.mean(0).argmax(-1)  # Taking the mean over all of the sample we've drawn
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    print('Test set: Accuracy: {}/{} ({}%)'.format(
        correct, len(test_loader.dataset), 100. * correct / float(len(test_loader.dataset))
    ))

import tqdm

for epoch in range(1, n_epochs + 1):
    with gpytorch.settings.use_toeplitz(False):
        train(epoch)
        test()
    scheduler.step()
    # state_dict = model.state_dict()
    # likelihood_state_dict = likelihood.state_dict()