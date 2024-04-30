import torch
import random

import matplotlib.pyplot as plt

from torch import optim
from torch.utils.data import DataLoader
from network import Img2Img, WassersteinLoss


class Trainer():
    def __init__(self, model: Img2Img, loss_fn: WassersteinLoss, optimizer: optim.RAdam, train_loader: DataLoader, test_data: torch.Tensor, device: torch.device) -> None:
        self.model          = model
        self.loss_fn        = loss_fn
        self.optimizer      = optimizer
        self.train_loader   = train_loader
        self.test_data      = test_data
        self.device         = device

    def train(self, epoches) -> None:
        # Set model to training model.
        self.model.train()

        for epoch in range(epoches):
            # Set defualt values.
            total_loss  = 0
            couter      = 0

            for x, y in self.train_loader:
                # Set defualt data type.
                x   : torch.Tensor
                y   : torch.Tensor
                loss: torch.Tensor

                # Update counter.
                couter += 1

                # Move data to device.
                x = x.to(self.device)
                y = y.to(self.device)

                # Get output from model.
                y_pred = self.model(x)

                # Calculate loss.
                loss = self.loss_fn(y_pred, y)
                total_loss += loss.item()

                # Update model.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Print training result.
                print(f"Epoch: {epoch} | Loss: {total_loss / couter}", end="\r")

            # Output testing result.
            self.test(10)

        # Set model to evaluation model.
        self.model.eval()

    @torch.no_grad()
    def test(self, test_num: int) -> None:
        # Set defualt figure.
        figure = plt.figure(figsize=(10, 10))

        for i in range(test_num):
            # Get sample data with specified number.
            sample = self.test_data[i % 10]

            # Get random data.
            r = random.randint(0, len(sample) - 1)
            x = sample[r]
            x = x.unsqueeze(0).to(self.device)

            # Get output image.
            y = self.model(x)

            # Show input image.
            axes = figure.add_subplot(10, 10, i + 1)
            axes.set_axis_off()
            plt.imshow(x.cpu().squeeze(), cmap="gray")

            # Show output image.
            axes = figure.add_subplot(10, 10, i + 11)
            axes.set_axis_off()
            plt.imshow(y.squeeze().detach().cpu(), cmap="gray")

        # Show figure.
        plt.show()