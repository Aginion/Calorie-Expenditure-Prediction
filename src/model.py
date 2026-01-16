import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, use_dropout: bool = True) -> None:
        super().__init__()

        dropout_layer = nn.Dropout(0.3) if use_dropout else nn.Identity()

        self.net = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            dropout_layer,

            nn.Linear(128, 64),
            nn.ReLU(),
            dropout_layer,

            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
