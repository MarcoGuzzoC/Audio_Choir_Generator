import numpy as np
import os
import pickle
from torch import nn
from GREGOConfig import GREGOConfig
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError


class ConvDown(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvDown, self).__init__()
        self.config = GREGOConfig()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=8,
            stride=1,
            padding=0,
            dilation=1,
        )
        self.norm = nn.BatchNorm2d(self.out_channels)
        self.dropout = nn.Dropout2d(self.config.dropout) #essayer de faire varier 
        self.fc = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.dropout(x)
        return self.fc(x)


class ConvUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvUp, self).__init__()
        self.config = GREGOConfig()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=8,
            stride=1,
            padding=0,
            dilation=1,
        )
        self.fc = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.config = GREGOConfig()
        self.in_channels = in_channels
        channels = [in_channels] + [2**i for i in range(3, 10)]
        self.encoder = nn.ModuleList(
            [ConvDown(channels[i], channels[i + 1]) for i in range(self.config.number_of_stack)]
        )

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels):
        super(Decoder, self).__init__()
        self.config = GREGOConfig()
        channels = [out_channels] + [2**i for i in range(3, 10)]
        self.decoder = nn.ModuleList(
            [ConvUp(channels[i + 1], channels[i]) for i in range(self.config.number_of_stack)]
            )[::-1]

    def forward(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.config = GREGOConfig()
        self.encoder = Encoder(self.in_channels, self.config.number_of_stack)
        self.decoder = Decoder(self.in_channels, self.config.number_of_stack)

    def forward(self, x):
        latent_representation = self.encoder(x)
        reconstructed_image = self.decoder(latent_representation)
        return reconstructed_image