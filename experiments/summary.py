from collections import defaultdict

import numpy as np

from multifidelity_gp.data_io import ROOT, TARGETS, read_json, write_json
from .protocol import METHODS


def build_summary():
    rows = {}
    for method in METHODS:
        groups = defaultdict(list)
        for path in sorted((ROOT / 'results' / 'ablation' / 'sc1095' / method).glob('*.json')):
            record = read_json(path)
            groups[record['split_seed']].append(record)
        scores = [float(np.mean([r['normalized_aulc'] for r in groups[seed]])) for seed in sorted(groups)]
        finals = [float(np.mean([r['states'][-1]['mean_nrmse'] for r in groups[seed]])) for seed in sorted(groups)]
        rows[method] = dict(run_count=sum(map(len, groups.values())), split_scores=scores,
                            mean_normalized_aulc=float(np.mean(scores)),
                            mean_final_nrmse=float(np.mean(finals)))
    full = rows['full_method']
    for method in METHODS[1:]:
        rows[method]['aulc_increase_percent'] = (
            rows[method]['mean_normalized_aulc'] / full['mean_normalized_aulc'] - 1) * 100
        rows[method]['full_method_reduction_percent'] = (
            1 - full['mean_normalized_aulc'] / rows[method]['mean_normalized_aulc']) * 100
        rows[method]['full_method_wins'] = sum(
            a < b for a, b in zip(full['split_scores'], rows[method]['split_scores']))
    fusion = {}
    for case in ('sc1095', 'a320', 'crm'):
        result = read_json(ROOT / 'results' / 'fusion' / case / 'fusion_results.json')
        fusion[case] = {key: result[key] for key in ('grid_count', 'initial_count', 'maximum_count', 'high_fidelity_fraction')}
        fusion[case]['untrained_nrmse'] = {target: result['states'][-1]['metrics'][target]['untrained']['nrmse']
                                         for target in TARGETS}
    budget = read_json(ROOT / 'results/fusion/budget_study/sc1095/budget_comparison.json')
    result = dict(ablation=rows, fusion=fusion, budget_study=budget)
    write_json(ROOT / 'reports' / 'summary.json', result)
    return result
