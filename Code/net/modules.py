# python script with all the modules for  the neural network structure
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import config as cfg
import numpy as np


# Double convolution for U-net ------------------------------------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# Contraction part of U-net only ------------------------------------------------------------------------------------------------------


class UNET_down(nn.Module):
    def __init__(
        self, in_channels=cfg.input_channels, features=cfg.features,
    ):
        super(UNET_down, self).__init__()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        return x, skip_connections


# Bottleneck part of U-net only
class UNET_bottleneck(nn.Module):
    def __init__(self, features=cfg.features):
        super(UNET_bottleneck, self).__init__()
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

    def forward(self, x, skip_connections):
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        return x, skip_connections

# Up-part of U-net only ---------------------------------------------------------------------------------------------------------------


class UNET_up(nn.Module):
    def __init__(self, out_channels=cfg.output_channels, features=cfg.features):
        super(UNET_up, self).__init__()
        self.ups = nn.ModuleList()

        curr = []
        for i, feature in enumerate(reversed(features)):
            curr.append(feature)
            if i == 0:
                self.ups.append(nn.ConvTranspose2d(
                    2*feature, feature, kernel_size=2, stride=2))
                self.ups.append(DoubleConv(2*feature, 2*feature))
                continue
            self.ups.append(nn.ConvTranspose2d(
                feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, skip_connections):
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

# Projection head for pretraining ------------------------------------------------------------------------------------------------------
# MLP with one hidden layer with output in 128-dimensional latent space


class Projection_head(nn.Module):
    def __init__(self, dims=cfg.dims):
        super(Projection_head, self).__init__()
        self.dims = dims

        self.MLP = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.dims[0], self.dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(self.dims[1], self.dims[2])
        )

    def forward(self, x):
        return self.MLP(x)

# Encoder part od U_net with projection head for contrastive learning ------------------------------------------------------------------


class PretrainEncoderProjectionHead(nn.Module):
    def __init__(self, in_channels=cfg.input_channels, features=cfg.features, dims=cfg.dims):
        super(PretrainEncoderProjectionHead, self).__init__()
        self.dims = dims
        self.downs = nn.ModuleList()

        self.MLP = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.dims[0], self.dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(self.dims[1], self.dims[2])
        )
        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            self.downs.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature

    def forward(self, x):

        skip_connections = []

        # Encoder
        for idx, down in enumerate(self.downs):
            x = down(x)
            if idx % 2 == 0:
                skip_connections.append(x)

        embedding = x

        # ProjectionHead
        projection = self.MLP(x)

        return embedding, projection, skip_connections


# full U-net to do heatmap classification/regression based on pretrained encoder part with contrastive loss ----------------------------
class U_net(nn.Module):
    def __init__(self, base_model, out_channels=cfg.output_channels, features=cfg.features, freeze_Encoder=False):
        super(U_net, self).__init__()
        self.out_channels = out_channels
        self.features = features
        self.freeze_Encoder = freeze_Encoder

        # Down part of UNET
        self.downs = base_model.downs

        # freeze embeddings for pretrained down part
        if freeze_Encoder:
            print("Freezing embeddings")
            for param in self.downs.parameters():
                param.requires_grad = False

        # Bottleneck of UNET
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Up part of UNET
        self.ups = nn.ModuleList()
        curr = []
        for i, feature in enumerate(reversed(features)):
            curr.append(feature)
            if i == 0:
                self.ups.append(nn.ConvTranspose2d(
                    2*feature, feature, kernel_size=2, stride=2))
                self.ups.append(DoubleConv(2*feature, 2*feature))
                continue
            self.ups.append(nn.ConvTranspose2d(
                feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        # Final convolution plus Sigmoid for Prediction in [0,1]
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []

        # Encoder
        for idx, down in enumerate(self.downs):
            x = down(x)
            if idx % 2 == 0:
                skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        # Final Operations / Activation
        x = self.final_conv(x)
        x = self.sig(x)

        return x

# full UNET for Baseline results (no pretrining etc.) ---------------------------------------------------------------------------------


class Baseline(nn.Module):
    def __init__(
        self, in_channels=cfg.input_channels, features=cfg.features, out_channels=cfg.output_channels
    ):
        super(Baseline, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            self.downs.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature

        # Up part of UNET
        curr = []
        for i, feature in enumerate(reversed(features)):
            curr.append(feature)
            if i == 0:
                self.ups.append(nn.ConvTranspose2d(
                    2*feature, feature, kernel_size=2, stride=2))
                self.ups.append(DoubleConv(2*feature, 2*feature))
                continue
            self.ups.append(nn.ConvTranspose2d(
                feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        # Bottleneck of UNET
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Final convolution plus Sigmoid for Prediction in [0,1]
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []

        # Encoder
        for idx, down in enumerate(self.downs):
            x = down(x)
            if idx % 2 == 0:
                skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        # Final Operations / Activation
        x = self.final_conv(x)
        x = self.sig(x)

        return x
