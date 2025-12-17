import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    
    def __init__(self) -> None:
        super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential
        (
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

