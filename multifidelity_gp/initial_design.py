import numpy as np

from .data_io import COORDINATES, TARGETS


def boundary_anchors(low):
    return [int(index) for group in low.groupby('mach', sort=True).indices.values()
            for index in (group[0], group[-1])]


def design_features(low):
    coordinates = low[COORDINATES].to_numpy(dtype=float)
    normalized = (coordinates - coordinates.mean(0)) / np.maximum(np.ptp(coordinates, axis=0) / 2, 1e-12)
    alpha, mach = normalized.T
    coefficients = low[list(TARGETS)].to_numpy(dtype=float).copy()
    coefficients[:, 1] = np.log(coefficients[:, 1])
    features = np.column_stack([alpha, mach, alpha ** 2, alpha * mach, mach ** 2, coefficients])
    features = (features - features.mean(0)) / np.maximum(features.std(0), 1e-12)
    return np.column_stack([np.ones(len(low)), features])


def d_optimal_design(low, candidates, count, ridge=1e-6):
    features = design_features(low)
    selected = boundary_anchors(low)
    information = np.eye(features.shape[1]) * ridge + features[selected].T @ features[selected]
    available = set(candidates) - set(selected)
    while len(selected) < count:
        options = sorted(available)
        phi = features[options]
        gains = np.einsum('ij,ji->i', phi, np.linalg.solve(information, phi.T))
        chosen = options[int(gains.argmax())]
        selected.append(chosen)
        available.remove(chosen)
        information += np.outer(features[chosen], features[chosen])
    return selected


def geometric_design(low, candidates, count):
    coordinates = low[COORDINATES].to_numpy()
    coordinates = (coordinates - coordinates.min(0)) / np.ptp(coordinates, axis=0)
    distances = ((coordinates[:, None] - coordinates[None, :]) ** 2).sum(2)
    selected = boundary_anchors(low)
    available = set(candidates) - set(selected)
    while len(selected) < count:
        options = sorted(available)
        chosen = options[int(distances[np.ix_(options, selected)].min(1).argmax())]
        selected.append(chosen)
        available.remove(chosen)
    return selected
