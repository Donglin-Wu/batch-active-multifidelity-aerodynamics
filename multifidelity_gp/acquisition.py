import numpy as np
import torch

from .data_io import COORDINATES, TARGETS


def integration_weights(low):
    weights = np.ones(len(low))
    for name in COORDINATES:
        knots = np.sort(low[name].unique())
        if len(knots) == 1:
            continue
        widths = np.empty_like(knots, dtype=float)
        widths[0] = (knots[1] - knots[0]) / 2
        widths[-1] = (knots[-1] - knots[-2]) / 2
        widths[1:-1] = (knots[2:] - knots[:-2]) / 2
        weights *= widths[np.searchsorted(knots, low[name].to_numpy())]
    return torch.tensor(weights / weights.sum(), dtype=torch.float64)


def variance_reduction_scores(covariances, noises, weights):
    denominator = covariances.diagonal(dim1=1, dim2=2).clamp_min(0) + noises
    return (weights[:, :, None] * covariances.square()).sum(1) / denominator


def condition_covariances(covariances, noises, chosen):
    for index, covariance in enumerate(covariances):
        column = covariance[:, chosen].clone()
        denominator = covariance[chosen, chosen].clamp_min(0) + noises[index, chosen]
        covariance -= torch.outer(column, column) / denominator
        covariance.copy_((covariance + covariance.T) / 2)


@torch.no_grad()
def select_batch(low, models, known, candidates, count=4, normalize=True,
                 condition_batch=True, random=False, rng=None):
    covariances, noises = [], []
    for target in TARGETS:
        inputs = torch.tensor(low[[*COORDINATES, target]].to_numpy(), dtype=torch.float64)
        covariances.append(models[target].posterior_covariance(inputs).clone())
        noises.append(models[target].observation_variance(inputs))
    covariances, noises = torch.stack(covariances), torch.stack(noises)
    weights = integration_weights(low)[None, :].repeat(len(TARGETS), 1) / len(TARGETS)
    if normalize:
        scales = torch.tensor(np.ptp(low[list(TARGETS)].to_numpy(), axis=0), dtype=torch.float64)
        weights = weights / scales[:, None].square()
    options = sorted(set(candidates) - set(known))
    before = float((weights * covariances.diagonal(dim1=1, dim2=2)).sum())
    selected, contributions = [], []
    fixed_scores = variance_reduction_scores(covariances, noises, weights)
    for _ in range(count):
        scores = (variance_reduction_scores(covariances, noises, weights)
                  if condition_batch else fixed_scores)
        chosen = int(rng.choice(options)) if random else options[int(scores.sum(0)[options].argmax())]
        selected.append(chosen)
        contributions.append(scores[:, chosen].tolist())
        options.remove(chosen)
        if condition_batch:
            condition_covariances(covariances, noises, chosen)
    if not condition_batch:
        for chosen in selected:
            condition_covariances(covariances, noises, chosen)
    after = float((weights * covariances.diagonal(dim1=1, dim2=2)).sum())
    record = dict(indices=selected, atomic_batch=True,
                  conditions_during_selection=condition_batch and not random)
    if not random:
        record.update(contributions=contributions,
                      expected_integrated_variance_before=before,
                      expected_integrated_variance_after=after,
                      expected_relative_variance_reduction=(before - after) / before)
    return selected, record
