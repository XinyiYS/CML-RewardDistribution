from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from data.pipeline import get_data_raw
from core.kernel import get_kernel
from core.reward_calculation import get_v


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class CustomView(nn.Module):  # Flattening layer for nn.Sequential
    def __init__(self, shape):
        super(CustomView, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class VisualizationCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        batch = next(iter(trainer.val_dataloaders[0]))
        images = batch[0][:32]
        images = images.to(pl_module.device)

        # Pass through autoencoder
        z = pl_module.encoder(images)
        x_hat = pl_module.decoder(z)

        images = torch.cat((images, x_hat), 0)
        images = images.cpu().detach().numpy()
        images = images * np.expand_dims(trainer.stds, axis=[1, 2]) + np.expand_dims(trainer.means, axis=[1, 2])
        pl_module.logger.experiment.add_image("Autoencoder reconstruction", images, pl_module.current_epoch, dataformats='NCHW')


class MMDCallback(Callback):
    def on_validation_end(self, trainer, pl_module):
        all_party_features = []
        for dataloader in trainer.party_dataloaders:
            ds_iter = iter(dataloader)
            party_features = []
            for batch in ds_iter:
                z = pl_module.encoder(batch[0].to(pl_module.device))
                party_features.append(z.cpu().detach().numpy())
            party_features = np.concatenate(party_features)
            all_party_features.append(party_features)
        all_party_features = np.array(all_party_features)

        ds_iter = iter(trainer.candidate_dataloader)
        candidate_features = []
        for batch in ds_iter:
            z = pl_module.encoder(batch[0].to(pl_module.device))
            candidate_features.append(z.cpu().detach().numpy())
        candidate_features = np.concatenate(candidate_features)

        v = get_v(all_party_features, candidate_features, pl_module.kernel, device=pl_module.device, batch_size=128)

        pl_module.logger.experiment.add_scalars('v', v, pl_module.current_epoch)


class LitAutoEncoder(pl.LightningModule):
    """
    LitAutoEncoder(
      (encoder): ...
      (decoder): ...
    )
    """

    def __init__(self, num_channels, side_dim, hidden_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512),
            CustomView((-1, 2048)),
            nn.Linear(2048, hidden_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            CustomView((-1, 512, 2, 2)),
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2,
                               output_padding=0 if side_dim == 28 else 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, num_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )

        self.kernel = get_kernel('rq', hidden_dim)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        loss = 0
        for dataset in batch:
            x, y = dataset
            z = self.encoder(x)
            x_hat = self.decoder(z)
            loss += F.mse_loss(x_hat, x)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--hidden_dim', default=16, type=int)
    parser.add_argument('--dataset', default='mnist', type=str)  # TODO: Remove default?
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--party_data_size', default=8000, type=int)
    parser.add_argument('--candidate_data_size', default=40000, type=int)
    parser.add_argument('--split', default='equaldisjoint', type=str)

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    # ------------
    # data
    # ------------
    party_datasets, party_labels, candidate_dataset, candidate_labels = get_data_raw(dataset=args.dataset,
                                                                                     num_classes=args.num_classes,
                                                                                     party_data_size=args.party_data_size,
                                                                                     candidate_data_size=args.candidate_data_size,
                                                                                     split=args.split)
    num_parties = len(party_datasets)
    num_channels = party_datasets.shape[-1]

    combined = np.concatenate([candidate_dataset, np.concatenate(party_datasets)])
    means = np.mean(combined.reshape(-1, num_channels), axis=0)
    stds = np.std(combined.reshape(-1, num_channels), axis=0)

    # Standardize data and turn into PyTorch datasets
    datasets = []
    party_dataloaders = []  # for MMD logging
    for i in range(num_parties):
        transformed = np.transpose((party_datasets[i] - means) / stds, (0, 3, 1, 2))
        dataset = TensorDataset(torch.tensor(transformed), torch.tensor(party_labels[i]))
        datasets.append(dataset)
        party_dataloaders.append(torch.utils.data.DataLoader(dataset,
                                                             batch_size=args.batch_size,
                                                             shuffle=False,
                                                             pin_memory=True))

    transformed = np.transpose((candidate_dataset - means) / stds, (0, 3, 1, 2))
    dataset = TensorDataset(torch.tensor(transformed), torch.tensor(candidate_labels))
    datasets.append(dataset)
    candidate_dataloader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=args.batch_size,
                                                       shuffle=False,
                                                       pin_memory=True)

    concat_dataset = ConcatDataset(*datasets)
    # train_loader returns a (num_parties + 1) length tuple, each element is a tuple with first element a tensor of
    # shape (batch_size, H, W, C) and second element a tensor of shape (batch_size, )
    train_loader = torch.utils.data.DataLoader(
        concat_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True)

    # Use combined dataset as validation dataloader
    transformed_combined = np.transpose((combined - means) / stds, (0, 3, 1, 2))
    combined_labels = np.concatenate([np.concatenate(party_labels), candidate_labels])
    combined_dataset = TensorDataset(torch.tensor(transformed_combined), torch.tensor(combined_labels))
    val_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True)

    # ------------
    # model
    # ------------
    if args.dataset == 'mnist':
        side_dim = 28
    elif args.dataset == 'cifar':
        side_dim = 32
    else:
        raise Exception("dataset argument needs to be either 'mnist' or 'cifar'")

    model = LitAutoEncoder(num_channels=num_channels,
                           side_dim=side_dim,
                           hidden_dim=args.hidden_dim)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # These pass into the VisualizationCallback
    trainer.means = means
    trainer.stds = stds
    trainer.callbacks.append(VisualizationCallback())

    # These are for MMDCallback
    trainer.party_dataloaders = party_dataloaders
    trainer.candidate_dataloader = candidate_dataloader
    trainer.callbacks.append(MMDCallback())

    trainer.callbacks.append(EarlyStopping(monitor='val_loss'))

    logger = TensorBoardLogger('lightning_logs', name='{}-{}-{}'.format(args.dataset,
                                                                        args.hidden_dim,
                                                                        args.split))
    trainer.logger = logger

    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    cli_main()
