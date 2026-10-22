import torch

from .data_io import COORDINATES, TARGETS
from .interpolation import LowFidelityMap


@torch.no_grad()
def predict_frame(models, low, coordinates):
    result = coordinates[COORDINATES].copy()
    inputs = torch.tensor(coordinates[COORDINATES].to_numpy(), dtype=torch.float64)
    for target in TARGETS:
        grid = torch.tensor(low[[*COORDINATES, target]].to_numpy(), dtype=torch.float64)
        baseline = LowFidelityMap(grid)(inputs[:, 0], inputs[:, 1])
        result[target] = models[target](torch.cat([inputs, baseline], 1)).flatten().numpy()
    return result
