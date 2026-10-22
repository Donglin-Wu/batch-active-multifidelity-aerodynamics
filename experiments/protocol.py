import numpy as np

from multifidelity_gp.data_io import ROOT, read_json


METHODS = ('full_method', 'without_residual', 'without_scale_normalization',
           'without_batch_conditioning', 'without_d_optimal', 'random_sampling')


def method_settings(method):
    return dict(response_mode='absolute' if method == 'without_residual' else 'residual',
                normalize=method != 'without_scale_normalization',
                condition_batch=method != 'without_batch_conditioning',
                geometric=method == 'without_d_optimal',
                random=method == 'random_sampling')


def ablation_protocol():
    return read_json(ROOT / 'config' / 'ablation.json')


def split_indices(low, seed):
    rng = np.random.default_rng(seed)
    test = []
    for indices in low.groupby('mach', sort=True).indices.values():
        for group in np.array_split(indices[1:-1], 4):
            test.extend(rng.choice(group, 4, replace=False).tolist())
    test = sorted(test)
    return test, sorted(set(range(len(low))) - set(test))
