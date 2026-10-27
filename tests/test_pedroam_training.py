import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pymap3d as pm
import torch

from pedroam import pipeline as p
from pedroam import training as t


class TrainingTests(unittest.TestCase):
    def test_scratch_initialization_uses_only_training_features_and_no_checkpoint(self):
        features = torch.tensor([[10., 1., 5.], [20., 1., 9.], [30., 1., 13.]], dtype=torch.float64)
        samples = [({'tensor_features': features[:2]}, torch.zeros(2)),
                   ({'tensor_features': features[2:]}, torch.zeros(2))]
        with patch.object(p, 'load_network', side_effect=AssertionError('Checkpoint loading forbidden')), \
                patch.object(torch, 'load', side_effect=AssertionError('Checkpoint loading forbidden')):
            torch.manual_seed(123)
            net, _ = t.initialize_network('scratch', samples)
            torch.manual_seed(123)
            repeat, _ = t.initialize_network('scratch', samples)
        torch.testing.assert_close(net.seq[0].mean, features.mean(dim=0))
        torch.testing.assert_close(net.seq[0].std, features.std(dim=0, unbiased=False).clamp_min(1e-6))
        self.assertTrue(all(torch.equal(v, repeat.state_dict()[k]) for k, v in net.state_dict().items()))
        self.assertFalse(net.seq[0].mean.requires_grad)
        self.assertFalse(net.seq[0].std.requires_grad)
        with self.assertRaisesRegex(ValueError, 'must not receive'):
            t.initialize_network('scratch', samples, 'not-allowed.pth')

    def test_projection_matches_scorer_and_has_correct_gradient(self):
        lat = np.array([22.3, 22.301, 22.29, -33., 65.])
        lon = np.array([114.18, 114.182, 114.2, 151., -20.])
        height = np.array([0., 100., -50., 500., 10000.])
        xyz = np.stack(pm.geodetic2ecef(lat, lon, height), axis=-1)
        value = torch.tensor(xyz, dtype=torch.float64, requires_grad=True)
        result = t.horizontal_from_ecef(value)
        np.testing.assert_allclose(result.detach().numpy(), p.horizontal_enu(lat, lon), atol=1e-7)
        point = value[1].detach().requires_grad_(True)
        self.assertTrue(torch.autograd.gradcheck(t.horizontal_from_ecef, (point,),
                                                eps=.01, atol=1e-6, rtol=1e-5))

    def test_scratch_comparison_handles_optional_baseline_and_common_population(self):
        entries = [{'wls_available': True, 'baseline_ecef': np.array(pm.geodetic2ecef(22.3, 114.18, 0))}
                   for _ in range(2)]
        streams = [{'meta': {'session': 'session', 'phone': 'P01'}, 'entries': entries,
                    'reference_valid': np.array([True, True]), 'truth_xy': np.zeros((2, 2))}]
        learned = pd.DataFrame({'time_utc_s': [100., 101.], 'latitude': [22.3, 22.3],
                                'longitude': [114.18, 114.18], 'available': [True, True]})
        official = learned.copy()
        official.loc[1, ['latitude', 'longitude']] = np.nan
        official.loc[1, 'available'] = False
        with tempfile.TemporaryDirectory() as tmp:
            rows = t.comparison(streams, [official], [learned], Path(tmp)/'with', 'scratch')
            self.assertEqual({r['method'] for r in rows}, {'wls', 'official', 'scratch'})
            common = [r for r in rows if r['population'] == 'common_epochs']
            self.assertTrue(all(r['epochs'] == 1 for r in common))
            self.assertEqual(next(r for r in common if r['method'] == 'scratch')['availability'], 1.)
            rows = t.comparison(streams, None, [learned], Path(tmp)/'without', 'scratch')
            self.assertEqual({r['method'] for r in rows}, {'wls', 'scratch'})
            self.assertTrue(all(r['epochs'] == 2 for r in rows))

    def test_session_leakage_and_unknown_sessions_are_rejected(self):
        manifest = {'streams': [{'session': s} for s in ['a', 'b', 'c']]}
        t.validate_splits(manifest, {'train': ['a'], 'validation': ['b'], 'test': ['c']})
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            t.validate_splits(manifest, {'train': ['a'], 'validation': ['a'], 'test': ['c']})
        with self.assertRaisesRegex(ValueError, 'unknown'):
            t.validate_splits(manifest, {'train': ['d'], 'validation': ['b'], 'test': ['c']})

    def test_validation_penalizes_missing_outputs_without_clipping_errors(self):
        streams = [{'reference_valid': np.array([True, True, False]),
                    'truth_xy': p.horizontal_enu(np.array([22.3]*3), np.array([114.18]*3))}]
        frame = pd.DataFrame({'latitude': [22.3, np.nan, 22.3],
                              'longitude': [114.18, np.nan, 114.18],
                              'available': [True, False, True]})
        stat = t.validation_stat(streams, [frame], 1000.)
        self.assertEqual(stat['selection_mean_m'], 500.)
        self.assertEqual(stat['failures'], 1)
        frame.loc[1, ['latitude', 'longitude', 'available']] = [23.3, 114.18, True]
        stat = t.validation_stat(streams, [frame], 1000.)
        self.assertGreater(stat['selection_mean_m'], 50000.)

    def test_gradient_accumulation_averages_samples_and_updates_partial_batch(self):
        net = torch.nn.Linear(1, 1, bias=False).double()
        with torch.no_grad():
            net.weight.fill_(2.)
        samples = [(torch.tensor([x], dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
                   for x in [1., 3., 2.]]

        class FixedOrder:
            def permutation(self, length):
                return np.arange(length)

        with patch.object(t, 'solve', side_effect=lambda model, entry: model(entry)), \
                patch.object(t, 'horizontal_from_ecef', side_effect=lambda value: value):
            stat = t.train_epoch(net, samples, torch.optim.SGD(net.parameters(), lr=.1),
                                 FixedOrder(), batch_size=2, grad_clip=100.)
        self.assertAlmostEqual(net.weight.item(), 1.6)
        self.assertEqual(stat['optimizer_steps'], 2)
        self.assertEqual(stat['trained_epochs'], 3)

    def test_solver_resets_initial_position_and_cleans_cache_on_failure(self):
        position = np.array([1., 2., 3.])
        entry = {'gnss': {'pos': position, 'cb': np.zeros(7), 'data': None,
                          'solve_data': None, 'raw_data': None},
                 'tensor_features': torch.zeros((4, 3), dtype=torch.float64)}
        net = lambda features: (torch.ones(4), torch.zeros(4))

        def solver(sample, nav, **kwargs):
            cached = t.tas.cache_data[id(sample)]
            np.testing.assert_array_equal(cached[0], [1., 2., 3.])
            cached[0][:] = 999.
            return {'status': True, 'pos': torch.ones(3)}

        with patch.object(t.tas, 'wls_pnt_pos', side_effect=solver):
            t.solve(net, entry)
            t.solve(net, entry)
        np.testing.assert_array_equal(position, [1., 2., 3.])
        self.assertNotIn(id(entry), t.tas.cache_data)
        with patch.object(t.tas, 'wls_pnt_pos', side_effect=ValueError('failed')):
            with self.assertRaisesRegex(ValueError, 'failed'):
                t.solve(net, entry)
        self.assertNotIn(id(entry), t.tas.cache_data)


if __name__ == '__main__':
    unittest.main()
