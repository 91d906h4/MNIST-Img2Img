import torch

from torch import nn


class Img2Img(nn.Module):
    def __init__(self) -> None:
        super(Img2Img, self).__init__()
        self.down1 = self.down(input_chennals=1, output_chennals=64, kernel_size=4, stride=2, padding=1)
        self.down2 = self.down(input_chennals=64, output_chennals=128, kernel_size=4, stride=2, padding=1)
        self.down3 = self.down(input_chennals=128, output_chennals=256, kernel_size=4, stride=2, padding=1)

        self.linear1 = nn.Linear(2304, 1024)
        self.linear2 = nn.Linear(1024, 2304)

        self.up1 = self.up(input_chennals=256, output_chennals=128, kernel_size=3, stride=2, padding=0)
        self.up2 = self.up(input_chennals=128, output_chennals=64, kernel_size=4, stride=2, padding=1)

        self.convt1 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

        self.flatten = nn.Flatten()

    @staticmethod
    def down(input_chennals: int, output_chennals: int, kernel_size: int, stride: int, padding: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels=input_chennals, out_channels=output_chennals, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(output_chennals),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def up(input_chennals: int, output_chennals: int, kernel_size: int, stride: int, padding: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=input_chennals, out_channels=output_chennals, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(output_chennals),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        x = self.flatten(x)

        x = self.linear1(x)
        x = self.linear2(x)

        x = x.view(-1, 256, 3, 3)

        x = self.up1(x)
        x = self.up2(x)

        x = self.convt1(x)

        return x


class WassersteinLoss(nn.Module):
    def __init__(self) -> None:
        super(WassersteinLoss, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)

        loss = torch.mean(torch.abs(x - y))

        return loss