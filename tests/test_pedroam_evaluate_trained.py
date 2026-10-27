from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from pedroam import evaluate_trained as e


class TrainedEvaluationTests(unittest.TestCase):
    def test_mean_is_epoch_weighted_and_common_population_is_shared(self):
        frame = pd.DataFrame({'evaluation_set': ['other_sessions']*4,
            'session': ['s']*4, 'phone': ['P01', 'P01', 'P02', 'P02'],
            'reference_eligible': [True, True, True, False],
            'wls': [4., 6., 8., 1000.], 'official': [2., np.nan, 12., 1000.],
            'model': [1., 3., 9., 1000.]})
        summary = e.summarize(frame, ['wls', 'official', 'model'])
        overall = summary[summary.scope == 'overall']
        available = overall[overall.population == 'all_available'].set_index('method')
        common = overall[overall.population == 'common_epochs'].set_index('method')
        self.assertAlmostEqual(available.loc['model', 'mean_m'], 13/3)
        self.assertEqual(common.loc['model', 'mean_m'], 5.)
        self.assertEqual(common.loc['official', 'mean_m'], 7.)
        self.assertTrue(common.epochs.eq(2).all())
        self.assertAlmostEqual(common.loc['official', 'availability'], 2/3)
        self.assertEqual(common.loc['model', 'availability'], 1.)

    def test_empty_reference_population_has_no_artificial_zero_error(self):
        frame = pd.DataFrame({'evaluation_set': ['heldout_tail'], 'session': ['s'], 'phone': ['p'],
                              'reference_eligible': [False], 'wls': [0.], 'model': [0.]})
        result = e.summarize(frame, ['wls', 'model'])
        self.assertTrue(result.mean_m.isna().all())
        self.assertTrue(result.availability.isna().all())
        self.assertTrue(result.epochs.eq(0).all())

    def test_archived_cutoff_is_applied_without_recomputing_or_mutating(self):
        streams = [{'meta': {'session': session, 'phone': 'p'},
                    'entries': [{'time_utc_s': x} for x in [1., 2., 3., 4.]]}
                   for session in ['training', 'other']]
        selected = e.select_streams(streams, 'training', 3., 'all-unseen')
        self.assertEqual([entry['time_utc_s'] for entry in selected[0]['entries']], [3., 4.])
        self.assertEqual(selected[0]['reference_time_bounds'], (3., np.inf))
        self.assertEqual(selected[1]['evaluation_set'], 'other_sessions')
        self.assertEqual(len(streams[0]['entries']), 4)
        self.assertEqual(len(e.select_streams(streams, 'training', 3., 'heldout-tail')), 1)
        self.assertEqual(e.select_streams(streams, 'training', 3., 'other-sessions')[0]['meta']['session'], 'other')

    def test_dataset_output_is_limited_to_results_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)/'dataset'
            self.assertEqual(e.check_output_path(dataset/'results/run', dataset), dataset/'results/run')
            for path in [dataset, dataset/'recordings/new-run', dataset/'reference/new-run']:
                with self.assertRaisesRegex(ValueError, 'results'):
                    e.check_output_path(path, dataset)

    def test_cache_input_revision_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            obs = dataset/'recordings/s/phones/p/gnsslogger/log_v302.26o'
            nav = dataset/'recordings/s/navigation/nav.rnx'
            obs.parent.mkdir(parents=True)
            nav.parent.mkdir(parents=True)
            obs.write_text('original observations')
            nav.write_text('navigation')
            meta = {'session': 's', 'phone': 'p', 'obs_sha256': e.p.sha256(obs),
                    'navigation_sha256': {'nav.rnx': e.p.sha256(nav)}}
            e.verify_cache_inputs(dataset, meta, {})
            obs.write_text('changed observations')
            with self.assertRaisesRegex(ValueError, 'rebuild-cache'):
                e.verify_cache_inputs(dataset, meta, {})


if __name__ == '__main__':
    unittest.main()
