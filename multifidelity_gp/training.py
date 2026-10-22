import time

import numpy as np
import torch

from .data_io import COORDINATES, TARGETS
from .model import MultifidelityGaussianProcess


def condition_fold_ids(grid, inputs, count=4):
    positions = [(inputs[:, axis:axis + 1] - grid[:, axis].unique(sorted=True)[None, :]).abs().argmin(1)
                 for axis in range(2)]
    return (positions[0] + 3 * positions[1]) % count


def fit_coefficient(low, high, indices, target, response_mode='residual', restarts=2):
    started = time.perf_counter()
    grid = torch.tensor(low[[*COORDINATES, target]].to_numpy(), dtype=torch.float64)
    features = torch.tensor(low[list(TARGETS)].to_numpy(), dtype=torch.float64)
    truth = torch.tensor(high.loc[indices, [target]].to_numpy(), dtype=torch.float64)
    inputs = grid[indices]
    folds = condition_fold_ids(grid, inputs)
    cv_predictions = torch.empty_like(truth)
    fold_records = []
    for fold in sorted(folds.unique().tolist()):
        keep = folds != fold
        fitted = MultifidelityGaussianProcess(grid, inputs[keep], truth[keep], features,
                                             target == 'cd', response_mode, restarts)
        with torch.no_grad():
            cv_predictions[~keep] = fitted(inputs[~keep])
        fold_records.append(dict(training_indices=np.asarray(indices)[keep.numpy()].tolist(),
                                 validation_indices=np.asarray(indices)[(~keep).numpy()].tolist()))
    model = MultifidelityGaussianProcess(grid, inputs, truth, features,
                                        target == 'cd', response_mode, restarts)
    record = dict(cv_predictions=cv_predictions.flatten().tolist(),
                  cv_truth=truth.flatten().tolist(),
                  cv_rmse=float((cv_predictions - truth).square().mean().sqrt()),
                  folds=fold_records, fit_seconds=time.perf_counter() - started,
                  response_mode=response_mode, logarithmic=target == 'cd',
                  parameters=dict(amplitude=model.base.amplitude, noise=model.base.noise,
                                  length_scales=model.base.length_scales.tolist(),
                                  response_center=float(model.base.response_center),
                                  response_scale=float(model.base.response_scale),
                                  evidence=model.base.evidence))
    return model, record


def fit_models(low, high, indices, response_mode='residual', restarts=2):
    models, records = {}, {}
    for target in TARGETS:
        models[target], records[target] = fit_coefficient(
            low, high, indices, target, response_mode, restarts)
    return models, records
