import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)  # Patch extraction
        self.layer2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)  # Non-linear mapping
        self.layer3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)   # Reconstruction

    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class RRDB(nn.Module):
    def __init__(self, channels=64, growth_channel=32):
        super(RRDB, self).__init__()

        self.conv1 = nn.Conv2d(channels, growth_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_channel, growth_channel, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_channel, growth_channel, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_channel, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x

        out = nn.functional.relu(self.conv1(x))
        out = nn.functional.relu(self.conv2(torch.cat([x, out], 1)))
        out = nn.functional.relu(self.conv3(torch.cat([x, out], 1)))
        out = self.conv4(torch.cat([x, out], 1))

        return out + residual

class Generator(nn.Module):
    def __init__(self):
        super(ESRGAN, self).__init__()
        self.initial_layer = nn.Conv2d(3, 64, kernel_size=3, padding=4)

        self.rrdb1 = RRDB()
        self.rrdb2 = RRDB()

        self.final_layer = nn.Conv2d(64, 3, kernel_size=3, padding=2)

    def forward(self, x):
        x = nn.functional.relu(self.initial_layer(x))

        x = self.rrdb1(x)
        x = self.rrdb2(x)

        x = self.final_layer(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, out_channels, normalization=True):
            layers = [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(3, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.AdaptiveAvgPool2d(1)
        )

        self.final_shit = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        out = torch.flatten(out, 1)
        out = self.final_shit(out)
        return out
