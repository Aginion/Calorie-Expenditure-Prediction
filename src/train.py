import os
import logging
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from .workout_dataset import WorkoutDataset
from .model import NeuralNetwork
from .rmsle_loss import RMSLELoss

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

logger: logging.Logger = logging.getLogger(__name__)


def train(train_loader, val_loader, model, epochs, lr, momentum):
    criterion = RMSLELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    best_val_rmsle = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                preds = model(X)
                val_loss += criterion(preds, y).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        logger.info(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train RMSLE: {train_loss:.4f} | Val RMSLE: {val_loss:.4f}"
        )

        if val_loss < best_val_rmsle:
            best_val_rmsle = val_loss
            save_path = os.path.join(HydraConfig.get().run.dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)

    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.legend()
    plt.show()


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    dataset = WorkoutDataset(cfg.data.train)

    generator = torch.Generator().manual_seed(SEED)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False
    )

    model = NeuralNetwork(use_dropout=True)

    train(
        train_loader,
        val_loader,
        model,
        cfg.training.epochs,
        cfg.training.learning_rate,
        cfg.training.momentum
    )


if __name__ == "__main__":
    main()
