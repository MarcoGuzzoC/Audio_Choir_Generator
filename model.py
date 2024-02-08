"""TODO"""

from torch import nn

from GREGOConfig import GREGOConfig


class ConvDown(nn.Module):
    """TODO"""

    def __init__(self, in_channels, out_channels):
        super(ConvDown, self).__init__()
        self.config = GREGOConfig()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
        )
        self.norm = nn.BatchNorm2d(self.out_channels)
        self.dropout = nn.Dropout2d(self.config.dropout)
        self.fc = nn.LeakyReLU(0.2)

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
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
        )
        self.fc = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, number_of_stack):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.number_of_stack = number_of_stack
        channels = [in_channels] + [2**i for i in range(3, 10)]
        self.encoder = nn.ModuleList(
            [ConvDown(channels[i], channels[i + 1]) for i in range(number_of_stack)]
        )

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels, number_of_stack):
        super(Decoder, self).__init__()
        channels = [out_channels] + [2**i for i in range(3, 10)]
        self.decoder = nn.ModuleList(
            [ConvUp(channels[i + 1], channels[i]) for i in range(number_of_stack)]
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
        # TODO : Define your forward
        latent_representation = self.encoder(x)
        reconstructed_image = self.decoder(latent_representation)
        return reconstructed_image
