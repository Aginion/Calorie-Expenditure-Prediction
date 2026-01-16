import torch
from torch.utils.data import Dataset
import pandas as pd


def load_data(csv_file: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = pd.read_csv(csv_file)

    # Encode categorical feature
    data["Sex"] = data["Sex"].map({"male": 1, "female": 0}).astype(float)

    features = data.drop(columns=["Calories", "id"])
    target = data["Calories"]

    x = torch.tensor(features.values, dtype=torch.float32)
    y = torch.tensor(target.values, dtype=torch.float32).view(-1, 1)

    return x, y


class WorkoutDataset(Dataset):
    def __init__(self, csv_file: str, normalize: bool = True) -> None:
        x, y = load_data(csv_file)

        if normalize:
            x_min, _ = x.min(dim=0, keepdim=True)
            x_max, _ = x.max(dim=0, keepdim=True)
            x = (x - x_min) / (x_max - x_min + 1e-8)

        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
