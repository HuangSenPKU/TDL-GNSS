"""From-scratch training on a session's first 80%, testing its final 20%."""
import argparse
import json
from pathlib import Path
import random
import shutil
import time

from . import training as t

np, pd, torch, p = t.np, t.pd, t.torch, t.p


def chronological_split(streams, train_fraction=.8):
    """Split pooled raw epochs before validity filtering; keep equal times together.

    A shared UTC cutoff prevents cross-phone temporal overlap. The first test
    timestamp defines the boundary; ties all enter test, so the training count
    can be slightly below floor(N * fraction).
    """
    if not 0 < train_fraction < 1:
        raise ValueError('Train fraction must lie strictly between zero and one')
    rows = []
    for stream in streams:
        times = np.array([e['time_utc_s'] for e in stream['entries']], dtype=float)
        if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('Each stream requires finite, strictly increasing timestamps')
        meta = stream['meta']
        rows.extend({'session': meta['session'], 'phone': meta['phone'], 'original_index': i,
                     'time_utc_s': float(timestamp)} for i, timestamp in enumerate(times))
    membership = pd.DataFrame(rows).sort_values(['time_utc_s', 'session', 'phone']).reset_index(drop=True)
    index = int(len(membership)*train_fraction)
    if index < 1 or index >= len(membership):
        raise ValueError('Need nonempty train and test splits')
    cutoff = float(membership.iloc[index].time_utc_s)
    membership['split'] = np.where(membership.time_utc_s < cutoff, 'train', 'test')
    training, testing = [], []
    for stream in streams:
        before = [e for e in stream['entries'] if e['time_utc_s'] < cutoff]
        after = [e for e in stream['entries'] if e['time_utc_s'] >= cutoff]
        if not before or not after:
            raise ValueError('Every phone must have epochs on both sides of the shared cutoff')
        training.append({'meta': stream['meta'].copy(), 'entries': before,
                         'reference_time_bounds': (-np.inf, cutoff)})
        testing.append({'meta': stream['meta'].copy(), 'entries': after,
                        'reference_time_bounds': (cutoff, np.inf)})
    return training, testing, membership, cutoff


def run(args):
    output, dataset, cache_root = args.output.resolve(), args.dataset.resolve(), args.cache_root.resolve()
    if output == dataset or dataset in output.parents:
        raise ValueError('Output must be outside the input dataset')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh output directory')
    manifest = json.loads((cache_root/'preprocess_manifest.json').read_text())
    if not manifest['complete'] or manifest['reference_used']:
        raise ValueError('Require complete GNSS-only caches')
    if args.session not in {s['session'] for s in manifest['streams']}:
        raise ValueError('Unknown session')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    streams = t.load_streams(cache_root, manifest, [args.session])
    training, testing, membership, cutoff = chronological_split(streams, args.train_fraction)
    output.mkdir(parents=True, exist_ok=True)
    source = output/'source'
    source.mkdir()
    for path in [Path(__file__), Path(t.__file__), Path(p.__file__),
                 Path(__file__).resolve().parents[1]/'model/model.py', Path(t.tas.__file__).parent/'core.py']:
        shutil.copy2(path, source/path.name)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(initialization='scratch', pretrained_parameters_loaded=False, validation_used=False,
                  selection_rule='fixed_epochs_final_checkpoint', model='HybridShareNet', device='cpu',
                  dtype='float64', normalization_source='eligible_training_observations',
                  loss='mean horizontal distance, zero-height WGS84 ENU',
                  split_basis='pooled raw observation epochs ordered by UTC; ties kept in test',
                  cutoff_utc_s=cutoff,
                  preprocess_manifest_sha256=p.sha256(cache_root/'preprocess_manifest.json'),
                  versions={name: p.version(name) for name in ['torch', 'numpy', 'pandas', 'pymap3d', 'tasgnss', 'pyrtklib']})
    membership.to_csv(output/'split_membership.csv', index=False)
    p.save_json(output/'config.json', config)
    p.save_json(output/'status.json', {'complete': False, 'stage': 'loading_training_references'})
    inventory = {'train': t.attach_references(training, dataset)}
    p.save_json(output/'data_inventory.json', inventory)
    samples = [(entry, torch.tensor(stream['truth_xy'][i], dtype=torch.float64))
               for stream in training for i, entry in enumerate(stream['entries'])
               if stream['reference_valid'][i] and entry['network_input_available']]
    if not samples:
        raise ValueError('No eligible training samples')
    net, _ = t.initialize_network('scratch', samples)
    normalization = {k: v.clone() for k, v in net.state_dict().items() if k.startswith('seq.0.')}
    torch.save(net.state_dict(), output/'initial.pth')
    p.save_json(output/'initialization.json', {
        'initialization': 'scratch', 'pretrained_parameters_loaded': False,
        'initial_state_sha256': p.sha256(output/'initial.pth'), 'seed': args.seed,
        'normalization_source': 'eligible_training_observations',
        'normalization': {k: v.tolist() for k, v in normalization.items()},
        'training_observations': sum(len(entry['tensor_features']) for entry, _ in samples)})
    print(f'Raw split: {membership.split.value_counts().to_dict()}; cutoff UTC={cutoff}', flush=True)
    print(f'Eligible training samples: {len(samples)}; fixed epochs: {args.epochs}', flush=True)
    optimizer = torch.optim.Adam((v for v in net.parameters() if v.requires_grad), lr=args.lr)
    history = []
    for epoch in range(1, args.epochs+1):
        start = time.monotonic()
        stat = t.train_epoch(net, samples, optimizer, rng, args.batch_size, args.grad_clip)
        if not all(torch.equal(net.state_dict()[k], v) for k, v in normalization.items()):
            raise RuntimeError('Normalization changed during training')
        torch.save(net.state_dict(), output/f'epoch_{epoch:03d}.pth')
        stat.update(epoch=epoch, elapsed_seconds=time.monotonic()-start)
        history.append(stat)
        p.save_json(output/'history.json', history)
        p.save_json(output/'status.json', {'complete': False, 'stage': 'training', 'epoch': epoch})
        print(json.dumps(stat), flush=True)
    # Fix the model before any test reference is attached or any test prediction is evaluated.
    shutil.copy2(output/f'epoch_{args.epochs:03d}.pth', output/'final.pth')
    p.save_json(output/'selection.json', {'selected_epoch': args.epochs,
        'selection_rule': 'fixed_epochs_final_checkpoint', 'validation_used': False,
        'test_used_for_selection': False, 'weights_sha256': p.sha256(output/'final.pth')})
    test_predictions = t.predict_streams(net, testing, output/'test_scratch')
    official = None
    if args.baseline_weights is not None:
        official_net, _ = p.load_network(args.baseline_weights)
        official = t.predict_streams(official_net, testing, output/'test_official')
    inventory['test'] = t.attach_references(testing, dataset)
    p.save_json(output/'data_inventory.json', inventory)
    summary = t.comparison(testing, official, test_predictions, output/'test_comparison', 'scratch')
    p.save_json(output/'status.json', {'complete': True, 'epochs_run': args.epochs,
        'selected_epoch': args.epochs, 'training_candidates': len(samples),
        'raw_training_epochs': int(membership.split.eq('train').sum()),
        'raw_test_epochs': int(membership.split.eq('test').sum()), 'test': summary})
    print('Final 20% test results:', flush=True)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--session', default='0421-AM-2026')
    parser.add_argument('--train-fraction', type=float, default=.8)
    parser.add_argument('--baseline-weights', type=Path, help='Optional post-training official benchmark')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--grad-clip', type=float, default=10.)
    parser.add_argument('--seed', type=int, default=20261027)
    args = parser.parse_args()
    if not 0 < args.train_fraction < 1:
        parser.error('train-fraction must lie strictly between zero and one')
    if not all(np.isfinite(v) and v > 0 for v in [args.epochs, args.batch_size, args.lr, args.grad_clip]):
        parser.error('Training hyperparameters must be finite and positive')
    run(args)


if __name__ == '__main__':
    main()
