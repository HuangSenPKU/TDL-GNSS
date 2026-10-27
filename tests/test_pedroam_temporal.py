from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from pedroam import temporal_training as temporal
from pedroam import training


def stream(phone, times):
    return {'meta': {'session': 'session', 'phone': phone},
            'entries': [{'time_utc_s': float(t), 'network_input_available': False} for t in times]}


class TemporalSplitTests(unittest.TestCase):
    def test_pooled_split_prevents_shifted_phone_time_overlap_before_filtering(self):
        streams = [stream('P01', np.arange(10)), stream('P02', np.arange(10)+2.3)]
        train, test, membership, cutoff = temporal.chronological_split(streams)
        self.assertEqual(len(membership), 20)
        self.assertEqual(membership.split.eq('train').sum(), 16)
        self.assertEqual([len(s['entries']) for s in train], [9, 7])
        before = [e['time_utc_s'] for s in train for e in s['entries']]
        after = [e['time_utc_s'] for s in test for e in s['entries']]
        self.assertLess(max(before), min(after))
        self.assertEqual(min(after), cutoff)
        self.assertFalse(any(e['network_input_available'] for s in train for e in s['entries']))

    def test_equal_timestamps_are_not_split_between_train_and_test(self):
        streams = [stream('P01', np.arange(10)), stream('P02', np.arange(10))]
        _, _, membership, cutoff = temporal.chronological_split(streams, .75)
        self.assertEqual(membership.split.eq('train').sum(), 14)
        self.assertEqual(cutoff, 7.)
        self.assertEqual(membership.loc[membership.time_utc_s.eq(7), 'split'].tolist(), ['test', 'test'])

    def test_invalid_timestamps_and_empty_partitions_are_rejected(self):
        for times in [[0., 0.], [1., 0.], [0., np.nan]]:
            with self.assertRaisesRegex(ValueError, 'strictly increasing'):
                temporal.chronological_split([stream('P01', times)])
        with self.assertRaisesRegex(ValueError, 'both sides'):
            temporal.chronological_split([stream('P01', [0., 1.]), stream('P02', [10., 11.])])

    def test_reference_matching_cannot_cross_temporal_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'recordings/session/reference/P01/trajectory-gt.csv'
            path.parent.mkdir(parents=True)
            for times, flags, query, bounds in [
                    ([99.8, 100.], [False, True], 99.99, (-np.inf, 100.)),
                    ([99.99, 100.2], [True, False], 100.01, (100., np.inf))]:
                pd.DataFrame({'observation_utc_seconds': times, 'epoch_id': [1, 2],
                              'gt_valid': flags, 'gt_latitude': [22.3, 22.3],
                              'gt_longitude': [114.18, 114.18]}).to_csv(path, index=False)
                current = stream('P01', [query])
                current['reference_time_bounds'] = bounds
                training.attach_references([current], Path(tmp))
                self.assertFalse(current['reference_valid'][0])


if __name__ == '__main__':
    unittest.main()
