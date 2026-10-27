"""Supervised PedROAM training, separate from the frozen benchmark CLI.

Run with ``python -m pedroam.training --help``. Only locally produced, trusted
GNSS caches may be loaded. Test references are opened after model selection.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import pandas as pd
import pymap3d as pm
import torch
import tasgnss as tas

from . import pipeline as p


def horizontal_from_ecef(position):
    """Differentiable WGS84 zero-height projection matching pipeline scoring.

    Fixed geodetic latitude iterations are for terrestrial GNSS positions;
    this is not a general conversion for positions near Earth's centre.
    """
    x, y, z = position.unbind(-1)
    rho = torch.sqrt(x*x + y*y)
    e2 = 6.6943799901413165e-3
    lat = torch.atan2(z, rho * (1-e2))
    for _ in range(8):
        radius = 6378137. / torch.sqrt(1-e2*torch.sin(lat)**2)
        lat = torch.atan2(z + e2*radius*torch.sin(lat), rho)
    lon = torch.atan2(y, x)
    radius = 6378137. / torch.sqrt(1-e2*torch.sin(lat)**2)
    surface = torch.stack((radius*torch.cos(lat)*torch.cos(lon),
                           radius*torch.cos(lat)*torch.sin(lon),
                           radius*(1-e2)*torch.sin(lat)), dim=-1)
    lat0, lon0 = np.radians([22.3, 114.18])
    origin = position.new_tensor(pm.geodetic2ecef(22.3, 114.18, 0))
    matrix = position.new_tensor([[-np.sin(lon0), np.cos(lon0), 0],
                                  [-np.sin(lat0)*np.cos(lon0),
                                   -np.sin(lat0)*np.sin(lon0), np.cos(lat0)]])
    return (surface-origin) @ matrix.T


def validate_splits(manifest, splits):
    known = {m['session'] for m in manifest['streams']}
    seen = set()
    for name, sessions in splits.items():
        if not sessions or len(set(sessions)) != len(sessions):
            raise ValueError(f'{name}: sessions must be nonempty and unique')
        if set(sessions) - known:
            raise ValueError(f'{name}: unknown sessions {set(sessions)-known}')
        if seen & set(sessions):
            raise ValueError('Train/validation/test sessions must be disjoint')
        seen.update(sessions)


def load_streams(cache_root, manifest, sessions):
    streams = []
    for meta in manifest['streams']:
        if meta['session'] not in sessions:
            continue
        path = cache_root / 'cache' / meta['session'] / meta['phone'] / 'gnss.pkl'
        if p.sha256(path) != meta['cache_sha256']:
            raise ValueError(f'Cache checksum mismatch: {path}')
        with path.open('rb') as handle:
            entries = pickle.load(handle)
        for entry in entries:
            if entry['network_input_available']:
                expected = p.features(entry['gnss'])
                np.testing.assert_array_equal(entry['features'], expected)
                entry['tensor_features'] = torch.tensor(expected, dtype=torch.float64)
        streams.append({'meta': meta, 'entries': entries})
    return streams


def attach_references(streams, dataset):
    inventory = []
    for stream in streams:
        meta = stream['meta']
        path = dataset / 'recordings' / meta['session'] / 'reference' / meta['phone'] / 'trajectory-gt.csv'
        references = pd.read_csv(path)
        if 'reference_time_bounds' in stream:
            lower, upper = stream['reference_time_bounds']
            references = references.loc[(references.observation_utc_seconds >= lower)
                                         & (references.observation_utc_seconds < upper)]
        rows, valid = p.match_reference(references,
                                        np.array([e['time_utc_s'] for e in stream['entries']]))
        stream['reference_valid'] = valid
        stream['truth_xy'] = p.horizontal_enu(rows.gt_latitude.to_numpy(), rows.gt_longitude.to_numpy())
        inventory.append({'session': meta['session'], 'phone': meta['phone'],
                          'reference_sha256': p.sha256(path), 'raw_epochs': len(valid),
                          'reference_eligible': int(valid.sum()),
                          'training_candidates': sum(bool(v and e['network_input_available'])
                                                     for v, e in zip(valid, stream['entries']))})
    return inventory


def initialize_network(initialization, samples, weights=None):
    """Scratch initialization never reads a checkpoint or external statistics."""
    source = Path(__file__).resolve().parents[1] / 'model' / 'model.py'
    if initialization == 'finetune':
        if weights is None:
            raise ValueError('Fine-tuning requires --weights')
        return p.load_network(weights)
    if initialization != 'scratch' or weights is not None:
        raise ValueError('Scratch training must not receive pretrained --weights')
    from model.model import HybridShareNet
    values = torch.cat([entry['tensor_features'] for entry, _ in samples], dim=0)
    if not torch.isfinite(values).all():
        raise ValueError('Nonfinite training features')
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False).clamp_min(1e-6)
    return HybridShareNet(mean, std).double(), source


def solve(net, entry):
    """Always restart from fixed GNSS-only WLS; never reuse learned positions."""
    sol, key = entry['gnss'], id(entry)
    tas.cache_data[key] = [np.asarray(sol['pos']).copy(), np.asarray(sol['cb']).copy(),
                           None, None, sol['data'], sol['solve_data'], sol['raw_data']]
    try:
        weight, bias = net(entry['tensor_features'])
        result = tas.wls_pnt_pos(entry, None, use_cache=True, w=weight, b=bias,
                                enable_torch=True, device='cpu')
        if not result['status'] or not torch.isfinite(result['pos']).all():
            raise ValueError(result.get('msg', 'Nonfinite learned position'))
        return result['pos'][:3]
    finally:
        tas.cache_data.pop(key, None)


def predict_streams(net, streams, folder):
    """Generate all positions without reading reference values."""
    folder.mkdir(parents=True, exist_ok=False)
    net.eval()
    predictions = []
    before = {k: v.clone() for k, v in net.state_dict().items()}
    with torch.no_grad():
        for stream in streams:
            meta = stream['meta']
            rows = []
            for entry in stream['entries']:
                row = {'time_utc_s': entry['time_utc_s'], 'available': False,
                       'latitude': np.nan, 'longitude': np.nan, 'failure_reason': ''}
                if entry['network_input_available']:
                    try:
                        position = solve(net, entry)
                        lat, lon, _ = pm.ecef2geodetic(*position.numpy())
                        if not np.isfinite([lat, lon]).all():
                            raise ValueError('Nonfinite coordinates')
                        row.update(available=True, latitude=lat, longitude=lon)
                    except (ValueError, RuntimeError) as error:
                        row['failure_reason'] = str(error)
                else:
                    row['failure_reason'] = entry.get('failure_reason', 'Input unavailable')
                rows.append(row)
            frame = pd.DataFrame(rows)
            path = folder / f"{meta['session']}__{meta['phone']}.csv"
            frame.to_csv(path, index=False)
            predictions.append(frame)
    if any(not torch.equal(before[k], v) for k, v in net.state_dict().items()):
        raise RuntimeError('Inference mutated network parameters')
    p.save_json(folder / 'manifest.json', {'complete': True, 'reference_used_for_prediction': False,
                                          'files': {f.name: p.sha256(f) for f in folder.glob('*.csv')}})
    return predictions


def errors_for(streams, predictions):
    errors, eligible = [], []
    for stream, frame in zip(streams, predictions):
        valid = stream['reference_valid']
        xy = p.horizontal_enu(frame.latitude.to_numpy(), frame.longitude.to_numpy())
        error = np.linalg.norm(xy-stream['truth_xy'], axis=1)
        error[~(valid & frame.available.to_numpy())] = np.nan
        errors.append(error)
        eligible.append(valid)
    return np.concatenate(errors), np.concatenate(eligible)


def validation_stat(streams, predictions, failure_penalty):
    errors, eligible = errors_for(streams, predictions)
    # Fixed denominator and explicit failure cost instead of dropping failures.
    objective = np.where(np.isfinite(errors), errors, failure_penalty)[eligible].mean()
    return {'selection_mean_m': float(objective), **p.metrics(errors),
            'eligible': int(eligible.sum()), 'failures': int((eligible & ~np.isfinite(errors)).sum())}


def train_epoch(net, samples, optimizer, rng, batch_size, grad_clip):
    net.train()
    total, accepted, failures, updates = 0., 0, Counter(), 0
    optimizer.zero_grad(set_to_none=True)
    pending = 0

    def update(count):
        for parameter in net.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(count)
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return float(norm)

    for index in rng.permutation(len(samples)):
        entry, target = samples[index]
        try:
            position = solve(net, entry)
            loss = torch.linalg.vector_norm(horizontal_from_ecef(position)-target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
        except (ValueError, RuntimeError) as error:
            failures[str(error)] += 1
            continue
        # Per-epoch backward bounds memory, accumulation averages actual samples.
        loss.backward()
        if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in net.parameters()):
            raise RuntimeError('Nonfinite gradient; stopping rather than updating corrupted weights')
        total += loss.item()
        accepted += 1
        pending += 1
        if pending == batch_size:
            update(pending)
            updates += 1
            pending = 0
        if accepted % 2000 == 0:
            print(f'  trained {accepted}/{len(samples)}; mean loss={total/accepted:.3f} m', flush=True)
    if pending:
        update(pending)
        updates += 1
    if not accepted:
        raise RuntimeError('No successful training samples')
    return {'train_mean_m': total/accepted, 'trained_epochs': accepted,
            'train_failures': dict(failures), 'optimizer_steps': updates}


def comparison(streams, official, trained, folder, learned_method='finetuned'):
    """Compare methods on both individual and identical populations."""
    folder.mkdir(parents=True, exist_ok=False)
    summary, frames = [], []
    for index, (stream, new) in enumerate(zip(streams, trained)):
        meta, entries, valid = stream['meta'], stream['entries'], stream['reference_valid']
        baseline = np.full((len(entries), 2), np.nan)
        for i, entry in enumerate(entries):
            if entry['wls_available']:
                lat, lon, _ = pm.ecef2geodetic(*entry['baseline_ecef'][:3])
                baseline[i] = p.horizontal_enu(lat, lon)
        coords = {'wls': baseline}
        if official is not None:
            old = official[index]
            coords['official'] = p.horizontal_enu(old.latitude.to_numpy(), old.longitude.to_numpy())
        coords[learned_method] = p.horizontal_enu(new.latitude.to_numpy(), new.longitude.to_numpy())
        frame = pd.DataFrame({'session': meta['session'], 'phone': meta['phone'],
                              'time_utc_s': new.time_utc_s, 'reference_eligible': valid})
        for name, xy in coords.items():
            error = np.linalg.norm(xy-stream['truth_xy'], axis=1)
            error[~valid] = np.nan
            frame[name] = error
        frame.to_csv(folder / f"{meta['session']}__{meta['phone']}.csv", index=False)
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    groups = [('all', 'all', all_rows)]
    groups.extend((s, phone, frame) for (s, phone), frame in all_rows.groupby(['session', 'phone']))
    for session, phone, frame in groups:
        eligible = frame.reference_eligible.to_numpy()
        common = eligible & np.isfinite(frame[list(coords)]).all(axis=1).to_numpy()
        for population, mask in [('all_available', eligible), ('common_epochs', common)]:
            for method in coords:
                summary.append({'session': session, 'phone': phone, 'population': population,
                                'method': method, 'eligible': int(eligible.sum()),
                                'availability': float(np.isfinite(frame[method]).sum()/eligible.sum()),
                                **p.metrics(frame.loc[mask, method])})
    pd.DataFrame(summary).to_csv(folder / 'summary.csv', index=False)
    return [row for row in summary if row['session'] == 'all']


def run(args):
    output, dataset, cache_root = args.output.resolve(), args.dataset.resolve(), args.cache_root.resolve()
    if output == dataset or dataset in output.parents:
        raise ValueError('Training output must be outside the input dataset')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Training requires a fresh output directory')
    manifest = json.loads((cache_root / 'preprocess_manifest.json').read_text())
    if not manifest['complete'] or manifest['reference_used']:
        raise ValueError('Require complete GNSS-only caches')
    splits = {'train': args.train_session, 'validation': args.validation_session, 'test': args.test_session}
    validate_splits(manifest, splits)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    model_source = Path(__file__).resolve().parents[1] / 'model' / 'model.py'
    output.mkdir(parents=True, exist_ok=True)
    source_folder = output / 'source'
    source_folder.mkdir()
    for source in [Path(__file__), Path(p.__file__), model_source,
                   Path(tas.__file__).parent / 'core.py']:
        shutil.copy2(source, source_folder / source.name)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(splits=splits, model='HybridShareNet', device='cpu', dtype='float64',
                  normalization_refitted=args.initialization == 'scratch',
                  normalization_source='training_samples' if args.initialization == 'scratch' else 'checkpoint',
                  loss='mean horizontal distance, zero-height WGS84 ENU',
                  initial_weights_sha256=p.sha256(args.weights) if args.weights is not None else None,
                  preprocess_manifest_sha256=p.sha256(cache_root / 'preprocess_manifest.json'),
                  versions={name: p.version(name) for name in ['torch', 'numpy', 'pandas', 'pymap3d', 'tasgnss', 'pyrtklib']})
    p.save_json(output / 'config.json', config)
    p.save_json(output / 'status.json', {'complete': False, 'stage': 'loading'})
    training = load_streams(cache_root, manifest, splits['train'])
    validation = load_streams(cache_root, manifest, splits['validation'])
    inventory = {'train': attach_references(training, dataset),
                 'validation': attach_references(validation, dataset)}
    p.save_json(output / 'data_inventory.json', inventory)
    samples = [(entry, torch.tensor(stream['truth_xy'][i], dtype=torch.float64))
               for stream in training for i, entry in enumerate(stream['entries'])
               if stream['reference_valid'][i] and entry['network_input_available']]
    if not samples or not any(s['reference_valid'].any() for s in validation):
        raise ValueError('Empty training or validation references')
    net, _ = initialize_network(args.initialization, samples, args.weights)
    normalization = {k: v.clone() for k, v in net.state_dict().items() if k.startswith('seq.0.')}
    torch.save(net.state_dict(), output / 'initial.pth')
    p.save_json(output / 'initialization.json', {
        'initialization': args.initialization, 'pretrained_parameters_loaded': args.initialization == 'finetune',
        'initial_state_sha256': p.sha256(output / 'initial.pth'), 'seed': args.seed,
        'normalization_source': config['normalization_source'],
        'normalization': {k: v.tolist() for k, v in normalization.items()},
        'training_observations': sum(len(entry['tensor_features']) for entry, _ in samples)})
    print(f'Training candidates: {len(samples)}', flush=True)
    initial_val = predict_streams(net, validation, output / 'validation_epoch_000')
    initial = validation_stat(validation, initial_val, args.failure_penalty)
    print(f'Initial validation: {initial}', flush=True)
    history = [{'epoch': 0, **initial}]
    best_score, best_epoch, stale = initial['selection_mean_m'], 0, 0
    torch.save(net.state_dict(), output / 'best.pth')
    optimizer = torch.optim.Adam((v for v in net.parameters() if v.requires_grad), lr=args.lr)
    for epoch in range(1, args.epochs+1):
        start = time.monotonic()
        stats = train_epoch(net, samples, optimizer, rng, args.batch_size, args.grad_clip)
        pred = predict_streams(net, validation, output / f'validation_epoch_{epoch:03d}')
        stats.update(validation_stat(validation, pred, args.failure_penalty))
        stats.update(epoch=epoch, elapsed_seconds=time.monotonic()-start)
        if not all(torch.equal(net.state_dict()[k], v) for k, v in normalization.items()):
            raise RuntimeError('Normalization changed during training')
        torch.save(net.state_dict(), output / f'epoch_{epoch:03d}.pth')
        if stats['selection_mean_m'] < best_score:
            best_score, best_epoch, stale = stats['selection_mean_m'], epoch, 0
            torch.save(net.state_dict(), output / 'best.pth')
        else:
            stale += 1
        history.append(stats)
        p.save_json(output / 'history.json', history)
        p.save_json(output / 'status.json', {'complete': False, 'stage': 'training',
                                           'epoch': epoch, 'best_epoch': best_epoch})
        print(json.dumps(stats), flush=True)
        if stale >= args.patience:
            print(f'Early stop after {stale} non-improving epochs', flush=True)
            break
    # Model choice is finalized and written before test references are opened.
    p.save_json(output / 'selection.json', {'best_epoch': best_epoch, 'best_validation_score': best_score,
                                          'weights_sha256': p.sha256(output / 'best.pth'),
                                          'test_used_for_selection': False})
    net.load_state_dict(torch.load(output / 'best.pth', map_location='cpu', weights_only=True), strict=True)
    best_val = [pd.read_csv(output / f'validation_epoch_{best_epoch:03d}' /
                           f"{s['meta']['session']}__{s['meta']['phone']}.csv") for s in validation]
    learned_method = 'scratch' if args.initialization == 'scratch' else 'finetuned'
    # Optional official weights are only used for post-selection benchmarking.
    baseline_weights = args.baseline_weights or args.weights
    official_net, official_val = None, None
    if baseline_weights is not None:
        official_net, _ = p.load_network(baseline_weights)
        official_val = predict_streams(official_net, validation, output / 'validation_official')
    val_summary = comparison(validation, official_val, best_val, output / 'validation_comparison', learned_method)
    # Release the large training caches before loading held-out sessions.
    del samples, training
    testing = load_streams(cache_root, manifest, splits['test'])
    test_trained = predict_streams(net, testing, output / f'test_{learned_method}')
    test_official = (predict_streams(official_net, testing, output / 'test_official')
                     if official_net is not None else None)
    inventory['test'] = attach_references(testing, dataset)
    p.save_json(output / 'data_inventory.json', inventory)
    test_summary = comparison(testing, test_official, test_trained, output / 'test_comparison', learned_method)
    p.save_json(output / 'status.json', {'complete': True, 'best_epoch': best_epoch,
                                       'epochs_run': history[-1]['epoch'],
                                       'training_candidates': sum(m['training_candidates'] for m in inventory['train']),
                                       'validation': val_summary, 'test': test_summary})
    print('Held-out test results:', flush=True)
    print(pd.DataFrame(test_summary).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--initialization', choices=['scratch', 'finetune'], default='finetune')
    parser.add_argument('--weights', type=Path, help='Training initialization checkpoint; fine-tuning only')
    parser.add_argument('--baseline-weights', type=Path, help='Optional official benchmark, loaded after selection')
    parser.add_argument('--output', type=Path, required=True)
    for name in ['train', 'validation', 'test']:
        parser.add_argument(f'--{name}-session', action='append', required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=10.)
    parser.add_argument('--failure-penalty', type=float, default=1000.)
    parser.add_argument('--seed', type=int, default=20261027)
    args = parser.parse_args()
    if args.initialization == 'scratch' and args.weights is not None:
        parser.error('Scratch mode forbids --weights; use --baseline-weights for post-training comparison')
    if args.initialization == 'finetune' and args.weights is None:
        parser.error('Fine-tuning requires --weights')
    if min(args.epochs, args.patience, args.batch_size, args.lr, args.grad_clip, args.failure_penalty) <= 0:
        parser.error('Training hyperparameters must be positive')
    run(args)


if __name__ == '__main__':
    main()
