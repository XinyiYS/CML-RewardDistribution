from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from data.pipeline import get_data_raw


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class VisualizationCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        batch = next(iter(trainer.train_dataloader))
        images = batch[-1][0][:32]
        images = images.to(pl_module.device)

        # Pass through autoencoder
        x = images.view(images.size(0), -1)
        z = pl_module.encoder(x)
        x_hat = pl_module.decoder(z)
        x_hat = x_hat.view(*images.size())

        images = torch.cat((images, x_hat), 0)
        images = images.cpu().detach().numpy()
        images = images * trainer.stds + trainer.means
        pl_module.logger.experiment.add_image("Topkeks", images, pl_module.current_epoch, dataformats='NHWC')


class LitAutoEncoder(pl.LightningModule):
    """
    LitAutoEncoder(
      (encoder): ...
      (decoder): ...
    )
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28),
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        loss = 0
        for dataset in batch:
            x, y = dataset
            x = x.view(x.size(0), -1)
            z = self.encoder(x)
            x_hat = self.decoder(z)
            loss += F.mse_loss(x_hat, x)
        return loss

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
    parser.add_argument('--party_data_size', default=4000, type=int)
    parser.add_argument('--candidate_data_size', default=10000, type=int)
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

    combined = np.concatenate([np.concatenate(party_datasets), candidate_dataset])
    means = np.mean(combined.reshape(-1, num_channels), axis=0)
    stds = np.std(combined.reshape(-1, num_channels), axis=0)

    # Standardize data and turn into PyTorch datasets
    datasets = []
    for i in range(num_parties):
        transformed = (party_datasets[i] - means) / stds
        dataset = TensorDataset(torch.tensor(transformed), torch.tensor(party_labels[i]))
        datasets.append(dataset)
    transformed = (candidate_dataset - means) / stds
    dataset = TensorDataset(torch.tensor(transformed), torch.tensor(candidate_labels))
    datasets.append(dataset)

    concat_dataset = ConcatDataset(*datasets)
    # loader returns a (num_parties + 1) length tuple, each element is a tuple with first element a tensor of shape
    # (batch_size, H, W, C) and second element a tensor of shape (batch_size, )
    loader = torch.utils.data.DataLoader(
        concat_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True)

    # ------------
    # model
    # ------------
    model = LitAutoEncoder(hidden_dim=args.hidden_dim)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.callbacks.append(VisualizationCallback())
    # These pass into the VisualizationCallback
    trainer.means = means
    trainer.stds = stds

    trainer.fit(model, train_dataloader=loader)

    # ------------
    # testing
    # ------------
    # result = trainer.test(test_dataloaders=loader)
    # print(result)


if __name__ == '__main__':
    cli_main()
