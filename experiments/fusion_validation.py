from pathlib import Path

import numpy as np
import torch

from multifidelity_gp.active_learning import FusionSession
from multifidelity_gp.data_io import COORDINATES, ROOT, TARGETS, load_data, read_json, write_json
from multifidelity_gp.model_io import save_models
from multifidelity_gp.prediction import predict_frame
from .metrics import regression_metrics


def run_case(case, output, maximum_count=None):
    torch.set_num_threads(1)
    config = read_json(ROOT / 'config' / 'fusion_validation.json')
    settings = config['cases'][case]
    if maximum_count is not None:
        settings['maximum_count'] = maximum_count
    low, high = load_data(case)
    directory = Path(output) / case
    directory.mkdir(parents=True, exist_ok=True)
    session = FusionSession(low, list(range(len(low))), settings['initial_count'],
                            settings['maximum_count'], config['batch_size'],
                            restarts=config['optimizer_restarts'])
    initial = session.pending.copy()
    responses = high.copy()
    responses[list(TARGETS)] = np.nan
    states = []
    scales = np.ptp(low[list(TARGETS)].to_numpy(), axis=0)
    while session.pending:
        responses.loc[session.pending, list(TARGETS)] = high.loc[session.pending, list(TARGETS)]
        training, batch = session.submit(responses)
        predictions = predict_frame(session.models, low, low)
        untrained = sorted(set(range(len(low))) - set(session.known))
        metrics = {}
        for target, scale in zip(TARGETS, scales):
            metrics[target] = {
                partition: regression_metrics(predictions.loc[indices, target].to_numpy(),
                                              high.loc[indices, target].to_numpy(),
                                              scale, target == 'cd')
                for partition, indices in [('all', list(range(len(low)))), ('untrained', untrained)]}
        state = dict(budget=len(session.known), training_indices=session.known.copy(),
                     training=training, metrics=metrics)
        if batch is not None:
            state['next_batch'] = batch
        states.append(state)
        print(f'{case}: completed {len(session.known)}/{settings["maximum_count"]} conditions', flush=True)
    record = dict(dataset=case, method='full_method', grid_count=len(low),
                  initial_count=len(initial), maximum_count=len(session.known),
                  high_fidelity_fraction=len(session.known) / len(low),
                  evaluation='all_and_untrained_grid_points', states=states)
    write_json(directory / 'fusion_results.json', record)
    low.loc[initial, COORDINATES].to_csv(directory / 'initial_points.csv', index=False)
    high.loc[session.known].to_csv(directory / 'used_high_fidelity.csv', index=False)
    predictions.to_csv(directory / 'predictions.csv', index=False)
    write_json(directory / 'metrics.json', states[-1]['metrics'])
    save_models(session.models, directory / 'model.pt')
    return str(directory)
