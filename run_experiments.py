import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from experiments.fusion_validation import run_case
from experiments.budget_study import compare_budgets
from experiments.protocol import METHODS, ablation_protocol
from experiments.runner import run_ablation
from experiments.summary import build_summary
from multifidelity_gp.data_io import ROOT


def main():
    parser = argparse.ArgumentParser(description='Reproduce multi-fidelity fusion and ablation experiments')
    commands = parser.add_subparsers(dest='command', required=True)
    ablation = commands.add_parser('ablation', help='Run the SC1095 ablation experiments')
    ablation.add_argument('--method', choices=(*METHODS, 'all'), default='all')
    ablation.add_argument('--seeds', nargs='+', type=int, default=ablation_protocol()['split_seeds'])
    ablation.add_argument('--workers', type=int, default=1)
    ablation.add_argument('--output', type=Path, default=ROOT / 'results' / 'ablation' / 'sc1095')
    fusion = commands.add_parser('fusion', help='Run fusion validation for the three datasets')
    fusion.add_argument('--case', choices=('sc1095', 'a320', 'crm', 'all'), default='all')
    fusion.add_argument('--workers', type=int, default=1)
    fusion.add_argument('--maximum-count', type=int, help='Set the final number of high-fidelity conditions')
    fusion.add_argument('--output', type=Path, default=ROOT / 'results' / 'fusion')
    commands.add_parser('summary', help='Update the experiment summary')
    budget = commands.add_parser('budget', help='Compare SC1095 budgets from 60 to 160 high-fidelity conditions')
    budget.add_argument('--trajectory', type=Path, help='Directory containing the 160-condition trajectory, predictions and model')
    args = parser.parse_args()
    if args.command == 'ablation':
        methods = METHODS if args.method == 'all' else (args.method,)
        tasks = [(method, seed, repeat) for method in methods for seed in args.seeds
                 for repeat in range(ablation_protocol()['random_repeats'] if method == 'random_sampling' else 1)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_ablation, *task, args.output) for task in tasks]
            for future in futures:
                future.result()
    elif args.command == 'fusion':
        cases = ('sc1095', 'a320', 'crm') if args.case == 'all' else (args.case,)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(partial(run_case, output=args.output, maximum_count=args.maximum_count), cases))
    elif args.command == 'budget':
        compare_budgets(args.trajectory)
    else:
        build_summary()


if __name__ == '__main__':
    main()
