# -*- coding: utf-8 -*-
"""
Created on Dec 21 14:57:02 2019
@author: Learning Deep Kernels for Two-sample Test
@Implementation of MMD-D and baselines in our paper on CIFAR dataset
BEFORE USING THIS CODE:
1. This code requires PyTorch 1.1.0, which can be found in
https://pytorch.org/get-started/previous-versions/ (CUDA version is 10.1).
2. This code also requires freqopttest repo (interpretable nonparametric two-sample test)
to implement ME and SCF tests, which can be installed by
   pip install git+https://github.com/wittawatj/interpretable-test
3. Numpy and Sklearn are also required. Users can install
Python via Anaconda (Python 3.7.3) to obtain both packages. Anaconda
can be found in https://www.anaconda.com/distribution/#download-section .
Note that runing time of ME test on CIFAR dataset is very long due to the huge dimension of data
"""
import argparse
import os
import sys
from os.path import join as oj
from collections import defaultdict

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
from torchsummary import summary

from utils.DK_tst_utils_HD import MatConvert, Pdist2, MMDu, TST_MMD_adaptive_bandwidth, TST_MMD_u

# Setup seeds
os.makedirs("images", exist_ok=True)
np.random.seed(819)
torch.manual_seed(819)
torch.cuda.manual_seed(819)
torch.backends.cudnn.deterministic = True
is_cuda = True

# parameters setting
parser = argparse.ArgumentParser()
parser.add_argument("--K", type=int, default=10, help="number of independent trials")
parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate for C2STs")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n", type=int, default=1000, help="number of samples in one set")
parser.add_argument("--dir", type=str, default='logs_dir', help="directory to store logs and figures")
parser.add_argument("--ARD", type=bool, default=True, help="separate lengscale hyperparameters")
opt = parser.parse_args()
print(opt)
dtype = torch.float
device = torch.device("cuda:0")
cuda = True if torch.cuda.is_available() else False
N_per = 100 # permutation times
alpha = 0.05 # test threshold
N1 = opt.n # number of samples in one set
K = opt.K # number of trails
N = 100 # number of test sets
N_f = 100.0 # number of test sets (float)

# Naming variables
ep_OPT = np.zeros([K])
s_OPT = np.zeros([K])
s0_OPT = np.zeros([K])
Results = np.zeros([6,K])

# Define the deep network for MMD-D
class Featurizer(nn.Module):
    def __init__(self):
        super(Featurizer, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0)] #0.25
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 300))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        feature = self.adv_layer(out)

        return feature



args = {}
args['dataset'] = 'CIFAR10'

n_participants = args['n_participants'] = 5
args['n_samples_per_participant'] = 2000
args['class_sz_per_participant'] = 2
args['n_samples'] = args['n_participants'] * args['n_samples_per_participant']

args['split_mode'] = "disjointclasses" #@param ["disjointclasses","uniform"," classimbalance", "powerlaw"]

args['include_joint'] = True
args['batch_size'] = opt.batch_size
args['num_channels'] = opt.channels if args['dataset'] == 'CIFAR10' else 1
args['image_size'] = opt.img_size if args['dataset'] == 'CIFAR10' else 28
args['ard_num_dims'] = opt.ARD

# fixed for MNIST and CIFAR10
args['num_classes'] = 10


import matplotlib.pyplot as plt
import seaborn as sns
import json

from utils.utils import evaluate, prepare_loaders, init_deterministic, setup_experiment_dir, setup_dir, write_model, tabulate_dict
from utils.mmd import mmd, t_statistic
joint_loader, train_loaders, joint_test_loader, test_loaders = prepare_loaders(args, repeat=args['include_joint'])

if args['include_joint']:
    train_loaders = [joint_loader] + train_loaders
    test_loaders = [joint_test_loader] + test_loaders

from itertools import product

N = args['n_participants'] + int(args['include_joint'])
pairs = list(product(range(N), range(N)))
# Repeat experiments K times (K = 10) and report average test power (rejection rate)
for kk in range(K):
    torch.manual_seed(kk * 19 + N1)
    torch.cuda.manual_seed(kk * 19 + N1)
    np.random.seed(seed=1102 * (kk + 10) + N1)
    # Initialize deep networks for MMD-D (called featurizer), C2ST-S and C2ST-L (called discriminator)
    featurizer = Featurizer()

    # Initialize parameters
    epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
    epsilonOPT.requires_grad = True
    sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * 32 * 32), device, dtype)
    sigmaOPT.requires_grad = True
    sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype)
    sigma0OPT.requires_grad = True
    # print(epsilonOPT.item())
    if cuda:
        featurizer = featurizer.to(torch.device('cuda'))
        if torch.torch.cuda.device_count() > 1:
            featurizer =  nn.DataParallel(featurizer, device_ids=list(range(torch.torch.cuda.device_count())))



    # Initialize optimizers
    optimizer = torch.optim.Adam(list(featurizer.parameters()) + [epsilonOPT] + [sigmaOPT] + [sigma0OPT], lr=0.0002)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------------------------------------------------------------------------------------------------
    #  Training deep networks for MMD-D (called featurizer)
    # ----------------------------------------------------------------------------------------------------
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    featurizer.train()
    for epoch in range(opt.n_epochs):
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
                XY = torch.cat([X, Y], 0)

                optimizer.zero_grad()
                
                # Compute output of deep network
                modelu_output = featurizer(XY)
                
                # Compute epsilon, sigma and sigma_0
                ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
                sigma = sigmaOPT ** 2
                sigma0_u = sigma0OPT ** 2

                # Compute Compute J (STAT_u)        
                if len(X) != len(Y):
                    continue
                    print("model output", modelu_output.shape, X.shape, XY.shape)        
                TEMP = MMDu(modelu_output, X.shape[0], XY.view(XY.shape[0],-1), sigma, sigma0_u, ep)
                mmd_value_temp = -1 * (TEMP[0])
                mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
                STAT_u = torch.div(mmd_value_temp, mmd_std_temp)

                if i != j:
                    mmd_losses[i] += STAT_u
                else:
                    mmd_losses[i] += -STAT_u

            loss = -torch.min(mmd_losses)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 100 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [Stat: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), -loss.item())
                )
        else:
            break

    print("Start evaluation:")

    featurizer.eval()
    mmd_dict = defaultdict(list)
    tstat_dict = defaultdict(list)
    with torch.no_grad():
        for data in zip(*test_loaders):
            data = list(data)
            for i in range(len(data)):
                data[i][0], data[i][1] = data[i][0].cuda(), data[i][1].cuda()

            for (i,j) in pairs:
                if i != j:
                    X, Y = data[i][0], data[j][0]
                else:
                    size = len(data[i][0])
                    temp = data[i][0]
                    rand_inds =  torch.randperm(size)
                    X, Y = temp[rand_inds[:size//2]], temp[rand_inds[size//2:]]

                if X.size(0) < 4 or Y.size(0) < 4: continue
                # Too small a batch leftover, would cause the t-statistic to be undefined, so skip
                XY = torch.cat([X, Y], 0)

                # Compute output of deep network
                modelu_output = featurizer(XY)
                
                # Compute epsilon, sigma and sigma_0
                ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
                sigma = sigmaOPT ** 2
                sigma0_u = sigma0OPT ** 2

                # Compute Compute J (STAT_u)        
                if len(X) != len(Y): continue

                TEMP = MMDu(modelu_output, X.shape[0], XY.view(XY.shape[0],-1), sigma, sigma0_u, ep)
                mmd_value_temp = -1 * (TEMP[0])
                mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
                STAT_u = torch.div(mmd_value_temp, mmd_std_temp)

                if i != j:
                    mmd_losses[i] += STAT_u
                else:
                    mmd_losses[i] += -STAT_u

                # mmd_hat, t_stat = featurizer(X, Y, pair=[i, j])
                mmd_dict[str(i)+'-'+str(j)].append(mmd_value_temp.tolist())
                tstat_dict[str(i)+'-'+str(j)].append(mmd_value_temp.tolist())

    os.makedirs(opt.dir, exist_ok=True)
    logdir = oj(opt.dir, 'trial-{}'.format(str(kk)))
    os.makedirs(logdir, exist_ok=True)

    with open(oj(logdir,'settings_dict'), 'w') as file:
        [file.write(key + ' : ' + str(value) + '\n') for key, value in args.items()]

    with open(oj(logdir,'opt_args.txt'), 'w') as file:
        json.dump(opt.__dict__, file, indent=2)

    with open(oj(logdir, 'model_summary.txt'), 'w') as file:
        sys.stdout = file # Change the standard output to the file we created.
        summary(featurizer, input_size=(opt.channels, opt.img_size, opt.img_size), batch_size=opt.batch_size)

    torch.save(featurizer.state_dict(), oj(logdir, 'featurizer.pth'))

    sys.stdout = open(os.path.join(logdir, 'log'), "w")
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

    with open(oj(logdir, 'mmd_dict'), 'w') as file:
        file.write(json.dumps(mmd_dict))
    
    with open(oj(logdir, 'tstat_dict'), 'w') as file:
        file.write(json.dumps(tstat_dict))

    figs_dir = oj(logdir, 'figs')
    mmd_dir = oj(figs_dir, 'mmd')
    tstat_dir = oj(figs_dir, 'tstat')
    os.makedirs(mmd_dir, exist_ok=True)
    os.makedirs(tstat_dir, exist_ok=True)

    from utils.plot import plot_together
    plot_together(mmd_dir, N=N, name='MMD', save=True, fig_dir=figs_dir)
    plot_together(tstat_dict, N=N, name='tStatistic', save=True, fig_dir=figs_dir)

    '''
    # plot and save MMD hats    
    for i in range(N):
        for j in range(N):
            pair = str(i)+'-'+str(j)
            mmd_values = np.asarray(mmd_dict[pair])*1e4
            sns.kdeplot(mmd_values, label=pair)

        plt.title('{} vs others MMD values'.format(str(i)))
        plt.xlabel('mmd values')
        # Set the y axis label of the current axis.
        plt.ylabel('density')
        # Set a title of the current axes.
        # show a legend on the plot
        plt.legend()
        # Display a figure.
        plt.savefig(oj(mmd_dir, '-'+str(i)))
        plt.clf()

    # plot and save Tstats  
    for i in range(N):
        for j in range(N):
            pair = str(i)+'-'+str(j)
            tstat_values = np.asarray(tstat_dict[pair])*1e4 // 10
            sns.kdeplot(tstat_values, label=pair)

        plt.title('{} vs others tstats values'.format(str(i)))
        plt.xlabel('tstat values')
        # Set the y axis label of the current axis.
        plt.ylabel('density')
        # Set a title of the current axes.
        # show a legend on the plot
        plt.legend()
        # Display a figure.
        plt.savefig(oj(tstat_dir, '-'+str(i)))
        plt.clf()



    # Run two-sample test on the training set
    # Fetch training data
    
    s1 = data_all[Ind_tr]
    s2 = data_trans[Ind_tr_v4]
    S = torch.cat([s1.cpu(), s2.cpu()], 0).cuda()
    Sv = S.view(2 * N1, -1)
    # Run two-sample test (MMD-D) on the training set
    h_u, threshold_u, mmd_value_u = TST_MMD_u(featurizer(S), N_per, N1, Sv, sigma, sigma0_u, ep, alpha, device, dtype)

    # Train MMD-O

    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)

    Dxy = Pdist2(Sv[:N1, :], Sv[N1:, :])
    sigma0 = Dxy.median()
    sigma0.requires_grad = True
    optimizer_sigma0 = torch.optim.Adam([sigma0], lr=0.0005)
    for t in range(opt.n_epochs):
        TEMPa = MMDu(Sv, N1, Sv, sigma, sigma0, is_smooth=False)
        mmd_value_tempa = -1 * (TEMPa[0] + 10 ** (-8))
        mmd_std_tempa = torch.sqrt(TEMPa[1] + 10 ** (-8))

        STAT_adaptive = torch.div(mmd_value_tempa, mmd_std_tempa)
        optimizer_sigma0.zero_grad()
        STAT_adaptive.backward(retain_graph=True)
        optimizer_sigma0.step()

        if t % 100 == 0:
            print("mmd: ", -1 * mmd_value_tempa.item(), "mmd_std: ", mmd_std_tempa.item(), "Statistic: ",
                  -1 * STAT_adaptive.item())
    h_adaptive, threshold_adaptive, mmd_value_adaptive = TST_MMD_adaptive_bandwidth(Sv, N_per, N1, Sv, sigma, sigma0, alpha, device, dtype)
    print("h:", h_adaptive, "Threshold:", threshold_adaptive, "MMD_value:", mmd_value_adaptive)

    '''
