from pathlib import Path

import numpy as np
import pandas as pd
import torch

from multifidelity_gp.data_io import COORDINATES, ROOT, TARGETS, load_data, read_json, write_json
from multifidelity_gp.model_io import load_models
from multifidelity_gp.prediction import predict_frame
from .metrics import regression_metrics


@torch.no_grad()
def compare_budgets(trajectory=None):
    torch.set_num_threads(1)
    directory = ROOT / 'results/fusion/budget_study/sc1095'
    trajectory = directory / 'budget_160' if trajectory is None else Path(trajectory)
    low, high = load_data('sc1095')
    record = read_json(trajectory / 'fusion_results.json')
    models = load_models(trajectory / 'model.pt')
    evaluation = sorted(set(low.index) - set(record['states'][-1]['training_indices']))
    scales = np.ptp(low[list(TARGETS)].to_numpy(), axis=0)
    rows, frames = [], []
    for state in record['states']:
        budget = state['budget']
        if budget not in range(60, 161, 20):
            continue
        known = state['training_indices']
        for target in TARGETS:
            model = models[target]
            base = model.base
            params = state['training'][target]['parameters']
            inputs = torch.tensor(low[[*COORDINATES, target]].to_numpy(), dtype=torch.float64)
            base.amplitude, base.noise = params['amplitude'], params['noise']
            for name in ('length_scales', 'response_center', 'response_scale'):
                setattr(base, name, inputs.new_tensor(params[name]))
            transformed = model.transform_inputs(inputs[known])
            base.anchor_features = base.extract_features(transformed)
            response = inputs.new_tensor(high.loc[known, [target]].to_numpy())
            response = response.log() if model.logarithmic else response
            residual = response - transformed[:, -1:]
            covariance = base.kernel(base.anchor_features, base.anchor_features)
            base.cholesky = torch.linalg.cholesky(
                covariance + base.noise * torch.eye(len(known), dtype=inputs.dtype))
            normalized = (residual - base.response_center) / base.response_scale
            base.coefficient = torch.cholesky_inverse(base.cholesky) @ normalized
            base.observation_variance = float(base.response_scale.square() * base.noise)
        prediction = predict_frame(models, low, low)
        targets = {}
        untrained = sorted(set(low.index) - set(known))
        for target, scale in zip(TARGETS, scales):
            actual = regression_metrics(prediction.loc[untrained, target],
                                        high.loc[untrained, target], scale, target == 'cd')
            np.testing.assert_allclose(actual['rmse'], state['metrics'][target]['untrained']['rmse'],
                                       rtol=1e-9, atol=1e-12)
            targets[target] = regression_metrics(prediction.loc[evaluation, target],
                                                 high.loc[evaluation, target], scale, target == 'cd')
        rows.append(dict(budget=budget, targets=targets))
        prediction.insert(0, 'budget', budget)
        prediction['evaluation'] = prediction.index.isin(evaluation)
        frames.append(prediction)
    write_json(directory / 'budget_comparison.json',
               dict(evaluation_count=len(evaluation), evaluation_indices=evaluation, states=rows))
    pd.concat(frames, ignore_index=True).to_csv(directory / 'budget_predictions.csv', index=False)
    return rows
