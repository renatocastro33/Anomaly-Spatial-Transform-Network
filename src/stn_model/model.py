import torch
from torch import Tensor, nn
import torch.nn.functional as F

class SpatialTransformerNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Calculate the size of the output from the convolutional layers
        self.conv_output_size = self._get_conv_output_size()

        self.fc_loc = nn.Sequential(
            nn.Linear(self.conv_output_size, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def _get_conv_output_size(self):
        # Utility function to calculate the output size from convolutional layers
        test_input = torch.randn(1, 3, 224, 224)
        test_output = self.localization(test_input)
        n_size = test_output.data.view(1, -1).size(1)
        return n_size

    def forward(self, x,mask=None):
        xs = self.localization(x)
        xs = xs.view(-1, self.conv_output_size)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = nn.functional.affine_grid(theta, x.size())

        x = nn.functional.grid_sample(x, grid)
        if not mask is None:
          return x, nn.functional.grid_sample(mask, grid)
        return x
        