import os
import urllib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
A Convolutional VAE for MNIST 28 by 28 with 

# 10-d latent space, for comparison with non-variational auto-encoder

'''

class Encoder(nn.Module):
    def __init__(self, latent_dims, ngf=64):
        super(Encoder, self).__init__()
        c = ngf
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1) # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
        self.fc_mu = nn.Linear(in_features=c*2*7*7, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=c*2*7*7, out_features=latent_dims)
            
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self, latent_dims, ndf=64):
        super(Decoder, self).__init__()
        self.c = c = ndf
        self.fc = nn.Linear(in_features=latent_dims, out_features=c*2*7*7)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.c*2, 7, 7) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims=10):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(latent_dims=latent_dims)
        self.decoder = Decoder(latent_dims=latent_dims)
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

def vae_loss(recon_x, x, mu, logvar, variational_beta = 10):
    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # Averaging or not averaging the binary cross-entropy over all pixels here
    # is a subtle detail with big effect on training, since it changes the weight
    # we need to pick for the other loss term by several orders of magnitude.
    # Not averaging is the direct implementation of the negative log likelihood,
    # but averaging makes the weight of the other loss term independent of the image resolution.
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    
    # KL-divergence between the prior distribution over latent vectors
    # (the one we are going to sample from when generating new images)
    # and the distribution estimated by the generator for the given image.
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + variational_beta * kldivergence

def load_pretrain(vae=None, path='MNIST_CVAE/vae_10d.pth'):
    vae = vae or VariationalAutoencoder()
    try:
        vae.load_state_dict(torch.load(path))
        print("Pretrained MNIST CVAE model <{}> loaded".format(path))
    except Exception as e:
        print(e)
        directory = 'MNIST_CVAE'
        filename = 'vae_10d.pth' #'vae_2d.pth'
        if not os.path.isdir(directory):
            os.makedirs(directory)
        print('downloading pretrained model ...')
        urllib.request.urlretrieve ("http://geometry.cs.ucl.ac.uk/creativeai/pretrained/"+filename, os.path.join(directory, filename))
        vae.load_state_dict(torch.load( os.path.join(directory,filename)))

        print("Pretrained MNIST CVAE model <{}> loaded".format(filename))
    return vae

if __name__=='__main__':
    vae = VariationalAutoencoder()
    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print('Number of parameters: %d' % num_params)
    vae = load_pretrain(vae)
