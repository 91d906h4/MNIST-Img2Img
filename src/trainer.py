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

        # Set defualt values.
        self.current_epoch  = 0

    def train(self, epoches) -> None:
        # Set model to training model.
        self.model.train()

        for _ in range(epoches):
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
                print(f"Epoch: {self.current_epoch} | Loss: {total_loss / couter}", end="\r")

            # Update current epoch.
            self.current_epoch += 1

            # Output testing result.
            self.test(test_num=10, postprocess=False)

        # Set model to evaluation model.
        self.model.eval()

    @torch.no_grad()
    def test(self, test_num: int, postprocess: bool=False, threshold: float=0.5) -> None:
        # Set defualt figure.
        figure = plt.figure(figsize=(10, 10))

        for i in range(test_num):
            # Get sample data with specified number.
            sample = self.test_data[i % 10]

            # Get random sample.
            r = random.randint(0, len(sample) - 1)
            x = sample[r]

            # Make input data to a batch with size 2.
            x = torch.stack([x, x])
            x = x.to(self.device)

            # Get output image.
            y = self.model(x)
            y = y[0]

            # Postprocess.
            # Replace pixel value to 0 if it's smaller than threshold.
            if postprocess:
                for r in range(28):
                    for c in range(28):
                        if y[0][r][c] <= threshold:
                            y[0][r][c] = 0

            # Show input image.
            axes = figure.add_subplot(10, 10, i + 1)
            axes.set_axis_off()
            plt.imshow(x[0].cpu().squeeze(), cmap="gray")

            # Show output image.
            axes = figure.add_subplot(10, 10, i + 11)
            axes.set_axis_off()
            plt.imshow(y.detach().cpu().squeeze(), cmap="gray")

        # Show figure.
        plt.show()