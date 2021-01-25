import torch
import torch.nn as nn
import torch.nn.functional as F

def MLP(args):
	if args['dataset'] == 'CIFAR10':
		return nn.Sequential(
			nn.Linear(args['num_features'], 256), 
			nn.Sigmoid(),
			nn.Linear(256, 128),
			nn.Sigmoid(),
			nn.Linear(128, 32)
			) 


	if args['dataset'] == 'MNIST':
		return nn.Sequential(
			nn.Linear(args['num_features'], args['num_features']), 
			nn.Sigmoid(), 
			nn.Linear(args['num_features'], args['num_features'])) 



# for MNIST 28*28
class MLP_MNIST(nn.Module):
	def __init__(self, in_dim=784, out_dim=10, device=None):
		super(MLP_MNIST, self).__init__()
		self.in_dim = in_dim
		self.fc1 = nn.Linear(self.in_dim, 512)
		self.fc2 = nn.Linear(512, 128)
		self.fc3 = nn.Linear(128, 64)
		self.fc4 = nn.Linear(64, out_dim)

	def forward(self, x):
		x = x.view(-1,  self.in_dim)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return x
		# return F.log_softmax(x, dim=1)

'''
class CNN_MNIST(nn.Module):
	def __init__(self, ngpu=1, nc=1, ndf=64, in_dim=784, out_dim=2, device=None):
		super(CNN_MNIST, self).__init__()
		self.ngpu = ngpu
		self.main = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
			nn.Sigmoid(),
			nn.Flatten(),
			nn.Linear(16, out_dim),
		)

	def forward(self, input):
		"""
		input: shape of (batch_size, c, h, w)
		"""
		if input.is_cuda and self.ngpu > 1:
			output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
		else:
			output = self.main(input)
		return output

# for MNIST 28*28
class MLP_MNIST(nn.Module):
	def __init__(self, in_dim=784, out_dim=2, device=None):
		super(MLP_MNIST, self).__init__()
		self.in_dim = in_dim
		self.fc1 = nn.Linear(self.in_dim, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(128, 64)
		self.fc4 = nn.Linear(64, out_dim)

	def forward(self, x):
		x = x.view(-1,  self.in_dim)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return x
		# return F.log_softmax(x, dim=1)
'''