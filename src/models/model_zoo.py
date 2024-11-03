import torch
from torch import Tensor
from torch import nn

from models.base import BaseModel


class Feature2DAirbusHelicopterAccelerometer(BaseModel):
    """
    A convolutional autoencoder designed for unsupervised anomaly detection on accelerometer data
    from the Airbus Helicopter Accelerometer Dataset. This model encodes input spectrogram patches
    into a latent bottleneck representation and then reconstructs them, aiming to identify abnormal
    patterns in accelerometer signals through reconstruction errors.

    Parameters:
        input_dim (int): Dimension of the input data (assumed to be square, i.e., input_dim x input_dim).
        encoder_name (str): Name of the encoder model architecture. Currently, only 'autoencoder_v1' is supported.
        bottleneck_dim (int, optional): Dimension of the bottleneck layer, used as the encoded feature size.
            Default is 300.
        variational (bool, optional): If True, adds variational components to the autoencoder by using
            reparameterization for encoding. Default is False.
        verbose (bool, optional): If True, enables detailed printouts during forward passes. Default is False.

    Attributes:
        encoder (nn.Sequential): The convolutional encoder network.
        decoder (nn.Sequential): The convolutional decoder network.
        fc_decode (nn.Sequential): Fully connected layers for decoding the bottleneck representation.
        fc_mu (nn.Sequential, optional): Layer for calculating the mean in the variational autoencoder,
            used if `variational` is True.
        fc_logvar (nn.Sequential, optional): Layer for calculating the log variance in the variational
            autoencoder, used if `variational` is True.

    Methods:
        print(msg): Prints messages when verbose mode is enabled.
        reparameterize(mu, log_var, is_train): Reparameterization trick for sampling during training
            in variational mode. Skips random sampling if `is_train` is False.
        forward(x: Tensor, is_train: bool = True): Passes input data through the encoder, bottleneck,
            and decoder, returning reconstructed data. If variational, also returns mean and log variance.

    Raises:
        ValueError: If an unsupported `encoder_name` is provided.

    Example usage:
        model = Feature2DAirbusHelicopterAccelerometer(
            input_dim=64,
            encoder_name="autoencoder_v1",
            bottleneck_dim=300,
            variational=True,
            verbose=True
        )
        reconstructed, mean, log_var = model(input_data, is_train=True)
    """
    def __init__(self, input_dim, encoder_name, bottleneck_dim=300, variational=False, verbose=False):
        super(Feature2DAirbusHelicopterAccelerometer, self).__init__()
        self.variational = variational
        self.bottleneck_dim = bottleneck_dim
        self.verbose = verbose

        activation = nn.LeakyReLU(negative_slope=0.01)

        if encoder_name == "autoencoder_v1":
            ks = 2
            self.nc_out = 128
            self.encoder = nn.Sequential(
                nn.Conv2d(1, self.nc_out // 4, kernel_size=ks, stride=2, padding=1),
                nn.BatchNorm2d(self.nc_out // 4),
                activation,
                nn.Conv2d(self.nc_out // 4, self.nc_out // 2, kernel_size=ks, stride=2, padding=1),
                nn.BatchNorm2d(self.nc_out // 2),
                activation,
                nn.Conv2d(self.nc_out // 2, self.nc_out, kernel_size=ks, stride=2, padding=1),
                nn.BatchNorm2d(self.nc_out),
                activation,
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(self.nc_out, self.nc_out // 2, kernel_size=ks, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(self.nc_out // 2),
                activation,
                nn.ConvTranspose2d(self.nc_out // 2, self.nc_out // 4, kernel_size=ks, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(self.nc_out // 4),
                activation,
                nn.ConvTranspose2d(self.nc_out // 4, 1, kernel_size=ks, stride=2, padding=1, output_padding=0),
                nn.Tanh()
            )

            n_layers, padding, stride, kernel_size = 3, 1, 2, ks
            self.nf_out = input_dim
            for _ in range(n_layers):
                self.nf_out = 1 + (self.nf_out + 2 * padding - kernel_size) // stride

        else:
            raise ValueError(f"Invalid encoder_name = {encoder_name}")

        self.fc_decode = nn.Sequential(
            nn.Linear(self.nc_out * self.nf_out**2, self.bottleneck_dim),
            activation,
            nn.Linear(self.bottleneck_dim, self.nc_out * self.nf_out ** 2),
            activation,
        )

        if self.variational:
            self.fc_mu = nn.Sequential(
                nn.Linear(self.nc_out * self.nf_out**2, self.bottleneck_dim),
                activation,
                nn.Linear(self.bottleneck_dim, self.nc_out * self.nf_out ** 2),
                activation,
            )
            self.fc_logvar = nn.Sequential(
                nn.Linear(self.nc_out * self.nf_out**2, self.bottleneck_dim),
                activation,
                nn.Linear(self.bottleneck_dim, self.nc_out * self.nf_out ** 2),
                activation,
            )

    def print(self, msg):
        if self.verbose:
            print(msg)

    def reparameterize(self, mu, log_var, is_train):
        if not is_train:
            x = mu
        else:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            x = mu + eps * std
        return x

    def forward(self, x: Tensor, is_train: bool = True):
        bs = x.shape[0]
        mean, log_var = None, None
        x = x.unsqueeze(1)
        self.print(f'1 - x.shape = {x.shape}')

        x = self.encoder(x)
        self.print(f'2 - x.shape = {x.shape}')

        x = x.view(bs, self.nc_out * self.nf_out ** 2)
        if not self.variational:
            x = self.fc_decode(x)
        else:
            mu = self.fc_mu(x)
            log_var = self.fc_logvar(x)
            x = self.reparameterize(mu, log_var, is_train)
            x = self.fc_decode(x)
        x = x.view(bs, self.nc_out, self.nf_out, self.nf_out)
        self.print(f'3 - x.shape = {x.shape}')

        x = self.decoder(x)
        self.print(f'4 - x.shape = {x.shape}')

        x = x.squeeze(1)

        return x, mean, log_var


if __name__ == '__main__':
    input_dim = 64
    nb, nwind, nfreq = 8, input_dim, input_dim
    feat = torch.randn(nb, nwind, nfreq)

    model = Feature2DAirbusHelicopterAccelerometer(
        input_dim=input_dim,
        encoder_name="autoencoder_v1",
        variational=True,
        verbose=True
    )
    print(model)

    reconst_feat, mean, log_var = model(feat)
    print(f"feat.shape = {feat.shape}")
    print(f"reconst_feat.shape = {reconst_feat.shape}")
    print(f"feat.min() .max() = {feat.min()}, {feat.max()}")
    print(f"reconst_feat.min() .max() = {reconst_feat.min()}, {reconst_feat.max()}")

    norm = feat.pow(2).sum((1, 2)).sqrt().mean(0)
    print(f"norm feat = {norm}")
    norm = reconst_feat.pow(2).sum((1, 2)).sqrt().mean(0)
    print(f"norm feat = {norm}")

    error = (reconst_feat - feat).pow(2).mean().sqrt()
    print(f"error = {error}")
