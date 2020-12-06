import argparse
import os
import numpy as np
import math
import copy

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from models.gan import Generator, Discriminator

# ----------
#  Argparse parameters
# ----------
'''
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--save_model",type=bool, default=False, help="whether to save models at the end of training")
parser.add_argument("--model_dir",type=str,default="saved_model", help="directory for saving the trained model")
opt = parser.parse_args()
img_shape = (opt.channels, opt.img_size, opt.img_size)
opt.img_shape = img_shape
'''


cuda = True if torch.cuda.is_available() else False
# ----------
#  Data Preparation
# ----------

# load the whole dataset
train = datasets.MNIST(
    ".data/mnist",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    ),
)

dataloader = torch.utils.data.DataLoader(
	train,
    batch_size=64,
    shuffle=True,
)

# determine the individual datasets based on labels/targets
n_parties = 10
Ds = []
for i in range(n_parties):
	idx = np.argwhere(np.asarray(train.targets==i))
	Ds.append(copy.deepcopy(train.data[idx]))

# ----------
#  Load pretrained models directly
# ----------

# Initialize generator and discriminator
from models.dcgan_mnist import Discriminator, Generator

D = Discriminator(ngpu=1).eval()
G = Generator(ngpu=1).eval()

# load weights
D.load_state_dict(torch.load('MNIST_DCGAN/netD_epoch_99.pth'))
G.load_state_dict(torch.load('MNIST_DCGAN/netG_epoch_99.pth'))
if cuda:
    D = D.cuda()
    G = G.cuda()
    # for d in Ds:
    	# d.cuda()

# ----------
#  MMD estimation
# ----------
from utils.utils import MMD_hat_squared, get_quantile, calculate_quantiles

# generate images from the trained generator
batch_size = 100
latent_size = 100
fixed_noise = torch.randn(batch_size, latent_size, 1, 1)
if cuda:
    fixed_noise = fixed_noise.cuda()
fake_images = G(fixed_noise)

# Data = copy.deepcopy(fake_images.data.cpu().numpy())
# quantiles = get_quantile(alpha=0.05, D=Data)


'''
generate the canditate set Candidates

for party i
	mu[i] <- compute estimates from Ds[i] 
	for each x in Candidates:
		delta[x] <- compute MMD(Ds[i] union x) - mu[i]
	priority queue the Candidates according to delta
	R[i] <- reward set
	while mu'[i] < mu[i]:
		added = False
		while not added:
		x = pq.pop(0)
		delta'[x] <- MMD(Ds[i] union R[i] union x) - mu'[i]
		if delta'[x] >= peek_max(pq):
			R[i].append(x)
			mu'[i] += delta'[x]
			added = True
		else:
			insert(G,x,delta'[x])
'''

# d1_G = torch.cat([Ds[0].float().cuda(), fake_images], dim=0)
# print(d1_G.shape)


# ----------
#  Greedy Algorithm
# ----------

import heapq

def greedy(mus, Ds, fake_images, canditates):	
	Rs = []
	for party in range(n_parties):
		data = Ds[party].float()
		pq = []
		mu_hat = MMD_hat_squared(data, fake_images)
		for x in canditates:
			x = torch.unsqueeze(x, 0).clone().detach()
			delta = MMD_hat_squared(torch.cat([data.cuda(), x], dim=0).cpu().numpy(), fake_images) - mu_hat
			delta += np.random.normal(0, 1e-12)
			heapq.heappush(pq, (-delta, x))

		R = torch.tensor([])
		while mu_hat < mus[party]:
			added = False
			while not added:
				x = heapq.heappop(pq)[1]
				delta = MMD_hat_squared(torch.cat([data.cuda(), R, x], dim=0).cpu().numpy(), fake_images) - mu_hat
				if delta >= pq[0][0]:
					R = torch.cat([R, x])
					mu_hat += delta
					added = True
				else:
					heapq.heappush(pq, (torch.tensor(-delta).cuda(), x))
		Rs.append(R)
	return Rs


def calculate_mus(rewards, fake_images, quantiles):
		
	'''
	max(rewards) leads to alpha = 0.01
	other rewards scale accordingly
	'''
	ALPHA = 0.01
	alphas = [ALPHA*max(rewards)/(reward) for reward in rewards]
	mus = [quantiles[int(len(quantiles)*alpha)] for alpha in alphas]
	# mu_hats = [MMD_hat_squared(data, fake_images) for data in Ds]
	return mus

rewards = np.ones(n_parties) / n_parties
rewards = (np.arange(n_parties)+1)/ n_parties

fake_images = copy.deepcopy(fake_images.data.cpu().numpy())
quantiles = calculate_quantiles(fake_images)
mus = calculate_mus(rewards, fake_images, quantiles)
mu_hats = [MMD_hat_squared(data, fake_images) for data in Ds]
print(rewards)
print(mus)
print(mu_hats)


# generate images from the trained generator
batch_size = 500
latent_size = 100
fixed_noise = torch.randn(batch_size, latent_size, 1, 1)
if cuda:
    fixed_noise = fixed_noise.cuda()
canditates = G(fixed_noise)
Rs = greedy(mus, Ds, fake_images, canditates)
