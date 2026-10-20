import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from pedroam import pipeline as p


class DatasetInterfaceTests(unittest.TestCase):
    def test_discover_without_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            obs = Path(tmp) / 'recordings/session/phones/P02/gnsslogger/example_v302.26o'
            obs.parent.mkdir(parents=True)
            obs.touch()
            self.assertEqual(p.discover(tmp), [('session', 'P02')])
            with self.assertRaises(ValueError):
                p.discover(tmp, ['missing/P01'])

    def test_three_feature_order_and_official_filter(self):
        data = np.zeros((4, 13), dtype=object)
        data[:, 2] = 'G'
        data[:, 9] = [30.9, 31.1, 32.4, 33.8]
        data[:, 12] = [1., 2., 3., 4.]
        sol = {'data': data, 'residual_info': {'residual': np.array([5., 6., 7., 8.])}}
        got = p.features(sol)
        np.testing.assert_array_equal(got[:, 0], [30, 31, 32, 33])
        np.testing.assert_array_equal(got[:, 1], [1, 2, 3, 4])
        np.testing.assert_array_equal(got[:, 2], [5, 6, 7, 8])
        # Vector norm, not a per-measurement 500 m threshold.
        sol['residual_info']['residual'] = np.full(4, 251.)
        with self.assertRaisesRegex(ValueError, '>500'):
            p.features(sol)

    def test_utc_matching_tolerance_and_invalid_reference(self):
        gt = pd.DataFrame({'observation_utc_seconds': [100., 101., 102.],
                           'epoch_id': [0, 1, 2], 'gt_valid': [True, False, True],
                           'gt_latitude': [22., 22., 22.], 'gt_longitude': [114., 114., 114.]})
        rows, valid = p.match_reference(gt, np.array([100.25, 100.26, 101., 102.]))
        np.testing.assert_array_equal(valid, [True, False, False, True])
        self.assertEqual(rows.epoch_id.tolist(), [0, 0, 1, 2])

    def test_projection_and_empty_metrics(self):
        np.testing.assert_allclose(p.horizontal_enu(np.array([22.3]), np.array([114.18])), 0)
        self.assertEqual(p.metrics([np.nan])['epochs'], 0)
        self.assertIsNone(p.metrics([])['rmse_m'])

    def test_scoring_requires_frozen_complete_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p.save_json(Path(tmp) / 'prediction_manifest.json', {'complete': False})
            with self.assertRaisesRegex(ValueError, 'complete'):
                p.score('/reference-must-not-be-opened', tmp)

    def test_score_counts_failure_without_error_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / 'dataset', Path(tmp) / 'run'
            folder = out / 'test/session/P02'
            folder.mkdir(parents=True)
            predictions = pd.DataFrame({'time_utc_s': [100., 101.], 'wls_available': [True, True],
                'tdl_available': [True, False], 'wls_latitude': [22.3, 23.3],
                'wls_longitude': [114.18, 114.18], 'tdl_latitude': [22.3, np.nan],
                'tdl_longitude': [114.18, np.nan], 'failure_reason': ['', 'rejected']})
            path = folder / 'predictions.csv'
            predictions.to_csv(path, index=False)
            p.save_json(out / 'prediction_manifest.json', {'complete': True, 'streams': [
                {'session': 'session', 'phone': 'P02', 'predictions_sha256': p.sha256(path)}]})
            gt = root / 'recordings/session/reference/P02/trajectory-gt.csv'
            gt.parent.mkdir(parents=True)
            pd.DataFrame({'observation_utc_seconds': [100., 101.], 'epoch_id': [0, 1],
                          'gt_valid': [True, True], 'gt_latitude': [22.3, 22.3],
                          'gt_longitude': [114.18, 114.18]}).to_csv(gt, index=False)
            p.score(root, out)
            d = pd.read_csv(out / 'summary.csv').set_index('method')
            self.assertEqual(d.loc['tdl', 'availability'], .5)
            self.assertGreater(d.loc['wls', 'rmse_m'], 10000)
            self.assertEqual(p.sha256(path), json.loads((out / 'prediction_manifest.json').read_text())['streams'][0]['predictions_sha256'])

    @unittest.skipUnless(os.getenv('TDL_TEST_WEIGHTS'), 'Set TDL_TEST_WEIGHTS for checkpoint loading test')
    def test_official_weights(self):
        net, _ = p.load_network(os.environ['TDL_TEST_WEIGHTS'])
        self.assertEqual(tuple(net.seq[1].weight.shape), (64, 3))
        self.assertFalse(net.training)


if __name__ == '__main__':
    unittest.main()
