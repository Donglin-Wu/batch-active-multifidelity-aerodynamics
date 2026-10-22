from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from experiments.fusion_validation import run_case
from experiments.protocol import split_indices
from multifidelity_gp.data_io import COORDINATES, TARGETS, load_data, read_json
from multifidelity_gp.initial_design import d_optimal_design
from multifidelity_gp.model_io import load_models
from multifidelity_gp.prediction import predict_frame


class WorkflowTests(unittest.TestCase):
    def test_paired_datasets(self):
        for case, count in [('sc1095', 405), ('a320', 28), ('crm', 28)]:
            low, high = load_data(case)
            self.assertEqual(len(low), count)
            self.assertEqual(len(high), count)
            np.testing.assert_array_equal(low[COORDINATES], high[COORDINATES])
            self.assertFalse(low.duplicated(COORDINATES).any())
            for frame in (low, high):
                self.assertTrue(np.isfinite(frame[list(TARGETS)].to_numpy()).all())
                self.assertTrue((frame.cd > 0).all())

    def test_initial_design_respects_test_split(self):
        low, _ = load_data('sc1095')
        for seed in range(10):
            test, candidates = split_indices(low, seed)
            initial = d_optimal_design(low, candidates, 20)
            self.assertEqual(len(test), 80)
            self.assertEqual(len(set(initial)), 20)
            self.assertFalse(set(initial).intersection(test))
            for indices in low.groupby('mach', sort=True).indices.values():
                self.assertIn(int(indices[0]), initial)
                self.assertIn(int(indices[-1]), initial)

    def test_a320_fusion_and_saved_model(self):
        torch.set_num_threads(1)
        low, high = load_data('a320')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(run_case('a320', temporary))
            result = read_json(directory / 'fusion_results.json')
            self.assertEqual([s['budget'] for s in result['states']], [10, 14])
            initial, final = result['states']
            selected = initial['next_batch']['indices']
            self.assertEqual(len(set(selected)), 4)
            self.assertTrue(initial['next_batch']['atomic_batch'])
            self.assertEqual(initial['training_indices'] + selected,
                             final['training_indices'])
            self.assertNotIn('next_batch', final)
            untrained = sorted(set(range(len(low))) - set(final['training_indices']))
            self.assertEqual(len(untrained), 14)
            for target in TARGETS:
                metric = final['metrics'][target]['untrained']
                baseline = np.sqrt(np.mean((low.loc[untrained, target] -
                                            high.loc[untrained, target]) ** 2))
                self.assertLess(metric['rmse'], baseline)
            model = load_models(directory / 'model.pt')
            actual = predict_frame(model, low, low)
            saved = pd.read_csv(directory / 'predictions.csv',
                                float_precision='round_trip')
            np.testing.assert_allclose(actual[list(TARGETS)], saved[list(TARGETS)],
                                       rtol=1e-12, atol=1e-12)
            self.assertTrue((actual.cd > 0).all())


if __name__ == '__main__':
    unittest.main()
