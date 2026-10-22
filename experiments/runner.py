from pathlib import Path
import time

import numpy as np
import torch

from multifidelity_gp.active_learning import FusionSession
from multifidelity_gp.data_io import COORDINATES, TARGETS, load_data, write_json
from multifidelity_gp.prediction import predict_frame
from .metrics import normalized_aulc, regression_metrics
from .protocol import ablation_protocol, method_settings, split_indices


def evaluate_models(models, low, high, indices, training_records):
    predictions = predict_frame(models, low, low.iloc[indices])
    scales = np.ptp(low[list(TARGETS)].to_numpy(), axis=0)
    for target, scale in zip(TARGETS, scales):
        predicted = predictions[target].to_numpy()
        truth = high.loc[indices, target].to_numpy()
        training_records[target].update(
            predictions=predicted.tolist(), truth=truth.tolist(),
            errors=(predicted - truth).tolist(),
            metrics=regression_metrics(predicted, truth, scale, target == 'cd'))
    return training_records


def run_ablation(method, seed, repeat, output):
    torch.set_num_threads(1)
    started = time.perf_counter()
    protocol = ablation_protocol()
    low, high = load_data('sc1095')
    test, candidates = split_indices(low, seed)
    random_seed = int(np.random.SeedSequence([seed, repeat, 20261020]).generate_state(1)[0])
    session = FusionSession(low, candidates, protocol['initial_count'],
                            protocol['maximum_count'], protocol['batch_size'],
                            rng=np.random.default_rng(random_seed),
                            restarts=protocol['optimizer_restarts'],
                            **method_settings(method))
    responses = high.copy()
    responses[list(TARGETS)] = np.nan
    record = dict(dataset='sc1095', method=method, split_seed=seed, repeat=repeat,
                  test_indices=test, test_coordinates=low.loc[test, COORDINATES].values.tolist(),
                  states=[])
    if method == 'random_sampling':
        record['random_seed'] = random_seed
    while session.pending:
        responses.loc[session.pending, list(TARGETS)] = high.loc[session.pending, list(TARGETS)]
        training, batch = session.submit(responses)
        targets = evaluate_models(session.models, low, high, test, training)
        state = dict(budget=len(session.known), training_indices=session.known.copy(),
                     targets=targets,
                     mean_nrmse=float(np.mean([targets[t]['metrics']['nrmse'] for t in TARGETS])))
        if batch is not None:
            state['next_batch'] = batch
        if record['states']:
            record['states'][-1]['next_batch']['actual_mean_nrmse_change'] = (
                state['mean_nrmse'] - record['states'][-1]['mean_nrmse'])
        record['states'].append(state)
    record.update(status='complete', normalized_aulc=normalized_aulc(record['states']),
                  elapsed_seconds=time.perf_counter() - started)
    path = Path(output) / method / f'split_{seed:02d}_repeat_{repeat:02d}.json'
    write_json(path, record)
    print(f'{method}: split {seed}, repeat {repeat} completed', flush=True)
    return str(path)
