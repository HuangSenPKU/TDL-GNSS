"""GNSS-only preprocessing, frozen inference and explicitly separate scoring."""
import hashlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

LEGACY_COMMIT = '76d9b684e1f7a60326514b1796ebfd829b532d46'
WEIGHT_URL = (f'https://raw.githubusercontent.com/ebhrz/TDL-GNSS/{LEGACY_COMMIT}/'
              'model/hybrid_share/hybrid_share.pth')
WEIGHT_SHA256 = '0bc1a4d2a6e214c3f92d8595f2ea65364873c57c0234f89cc6546be83594161e'
SYSTEMS = {'G': 0, 'R': 1, 'E': 2, 'C': 3, 'J': 4}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def discover(dataset, selected=None):
    """Discover phone streams from RINEX, without opening reference files."""
    root = Path(dataset) / 'recordings'
    found = sorted(root.glob('*/phones/*/gnsslogger/*_v302.26o'))
    ids = [(p.parents[3].name, p.parents[1].name) for p in found]
    if len(set(ids)) != len(ids):
        raise ValueError('Multiple converted observation files for the same phone stream')
    wanted = set(selected or [])
    if wanted - {f'{s}/{p}' for s, p in ids}:
        raise ValueError(f'Unknown recordings: {sorted(wanted - {f"{s}/{p}" for s, p in ids})}')
    result = [pair for pair in ids if not wanted or '/'.join(pair) in wanted]
    if not result:
        raise ValueError('No recordings/*/phones/*/gnsslogger/*_v302.26o files found')
    return result


def features(solution):
    """Match official feature quantization/filter; three-input feature order."""
    data = solution['data']
    if any(s not in SYSTEMS for s in data[:, 2]):
        raise ValueError('Constellation outside official GRECJ systems')
    residual = np.asarray(solution['residual_info']['residual']).reshape(-1)
    if np.linalg.norm(residual) > 500:
        raise ValueError('Official input residual-norm >500 m rejection')
    # No standardization here: original training statistics live in the checkpoint.
    values = np.column_stack([data[:, 9].astype(np.int8), data[:, 12].astype(float), residual])
    if not np.isfinite(values).all() or len(values) < 4:
        raise ValueError('Invalid network features')
    return values


def prepare(dataset, output, selected=None, limit_epochs=None):
    import tasgnss as tas
    root, out = Path(dataset), Path(output)
    streams = discover(root, selected)
    if (out / 'preprocess_manifest.json').exists():
        raise FileExistsError('Preprocessing manifest already exists; use a fresh output directory')
    manifest = {'complete': False, 'streams': [], 'epoch_stride': 1,
                'reference_used': False, 'limit_epochs': limit_epochs,
                'backend': {p: version(p) for p in ['tasgnss', 'pyrtklib', 'numpy']},
                'utc_conversion': 'TASGNSS obs2utc, GPST minus 18 s for April 2026; not universal'}
    save_json(out / 'preprocess_manifest.json', manifest)
    for session, phone in streams:
        cache = out / 'cache' / session / phone / 'gnss.pkl'
        if cache.exists():
            raise FileExistsError(f'Refusing to overwrite cache: {cache}; use a fresh output directory')
        recording = root / 'recordings' / session
        obs_paths = sorted((recording / 'phones' / phone / 'gnsslogger').glob('*_v302.26o'))
        nav_paths = sorted((recording / 'navigation').glob('*.rnx'))
        if len(obs_paths) != 1 or not nav_paths:
            raise ValueError(f'Expected one converted observation and navigation files: {session}/{phone}')
        obs, nav, _ = tas.read_obs(str(obs_paths[0]), [str(p) for p in nav_paths], opt='-SYS=GRECJ')
        epochs = tas.split_obs(obs)
        selected_epochs = epochs if limit_epochs is None else epochs[:limit_epochs]
        entries = []
        for index, epoch in enumerate(selected_epochs):
            entry = {'time_utc_s': float(tas.obs2utc(epoch.data[0].time)),
                     'wls_available': False, 'network_input_available': False}
            try:
                solution = tas.wls_pnt_pos(epoch, nav, return_residual=True, w=1)
                if solution['status'] and np.isfinite(solution['pos']).all():
                    entry.update(wls_available=True, baseline_ecef=np.asarray(solution['pos']).copy())
                    entry['features'] = features(solution)
                    entry.update(gnss=solution, network_input_available=True)
                else:
                    entry['failure_reason'] = solution.get('msg', 'Initial WLS failed')
            except Exception as error:
                entry['failure_reason'] = str(error)
            finally:
                tas.cache_data.clear()
            entries.append(entry)
            if (index + 1) % 500 == 0:
                print(f'prepare {session}/{phone}: {index+1}/{len(selected_epochs)}', flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open('wb') as f:
            pickle.dump(entries, f, protocol=4)
        metadata = {'session': session, 'phone': phone, 'raw_epochs': len(epochs),
                    'selected_epochs': len(entries), 'epoch_stride': 1, 'reference_used': False,
                    'obs_sha256': sha256(obs_paths[0]),
                    'navigation_sha256': {p.name: sha256(p) for p in nav_paths},
                    'cache_sha256': sha256(cache)}
        save_json(cache.with_suffix('.json'), metadata)
        manifest['streams'].append(metadata)
        save_json(out / 'preprocess_manifest.json', manifest)
        print(f'prepared {session}/{phone}: {len(entries)} epochs', flush=True)
    manifest['complete'] = True
    save_json(out / 'preprocess_manifest.json', manifest)


def load_network(weights):
    import torch
    if sha256(weights) != WEIGHT_SHA256:
        raise ValueError('Weight SHA256 differs from the pinned official pretrained checkpoint')
    source = Path(__file__).resolve().parents[1] / 'model' / 'model.py'
    spec = importlib.util.spec_from_file_location('official_model', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.HybridShareNet().double()
    net.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    net.eval()
    return net, source


def predict(output, weights):
    import pymap3d as pm
    import tasgnss as tas
    import torch
    out = Path(output)
    if (out / 'prediction_manifest.json').exists():
        raise FileExistsError('Prediction manifest already exists; use a fresh output directory')
    manifest = json.loads((out / 'preprocess_manifest.json').read_text())
    if not manifest['complete']:
        raise ValueError('Preprocessing incomplete')
    net, source = load_network(weights)
    before = {k: v.clone() for k, v in net.state_dict().items()}
    run = {'complete': False, 'training': False, 'reference_used': False,
           'variant': 'current HybridShareNet + official legacy hybrid_share.pth',
           'weights_sha256': sha256(weights), 'model_source_sha256': sha256(source),
           'normalization_refitted': False, 'streams': [],
           'versions': {p: version(p) for p in ['torch', 'torchvision', 'tasgnss', 'pyrtklib']}}
    save_json(out / 'prediction_manifest.json', run)
    with torch.no_grad():
        for metadata in manifest['streams']:
            session, phone = metadata['session'], metadata['phone']
            target = out / 'test' / session / phone / 'predictions.csv'
            if target.exists():
                raise FileExistsError(f'Refusing to overwrite predictions: {target}')
            cache = out / 'cache' / session / phone / 'gnss.pkl'
            if sha256(cache) != metadata['cache_sha256']:
                raise ValueError(f'Cache checksum mismatch: {cache}')
            # Only load caches produced locally by prepare; pickle is not safe for untrusted input.
            with cache.open('rb') as f:
                entries = pickle.load(f)
            rows = []
            for entry in entries:
                row = {'time_utc_s': entry['time_utc_s'], 'wls_available': entry['wls_available'],
                       'tdl_available': False, 'wls_latitude': np.nan, 'wls_longitude': np.nan,
                       'tdl_latitude': np.nan, 'tdl_longitude': np.nan, 'failure_reason': ''}
                if entry['wls_available']:
                    lat, lon, _ = pm.ecef2geodetic(*entry['baseline_ecef'][:3])
                    row.update(wls_latitude=lat, wls_longitude=lon)
                if entry['network_input_available']:
                    key = id(entry)
                    sol = entry['gnss']
                    tas.cache_data[key] = [np.asarray(sol['pos']).copy(), np.asarray(sol['cb']).copy(),
                                           None, None, sol['data'], sol['solve_data'], sol['raw_data']]
                    try:
                        weight, bias = net(torch.tensor(entry['features'], dtype=torch.float64))
                        result = tas.wls_pnt_pos(entry, None, use_cache=True, return_residual=False,
                                                w=weight, b=bias, enable_torch=True, device='cpu')
                        if not result['status'] or not torch.isfinite(result['pos']).all():
                            raise ValueError(result.get('msg', 'Learned WLS failed'))
                        lat, lon, _ = pm.ecef2geodetic(*result['pos'].numpy()[:3])
                        row.update(tdl_available=True, tdl_latitude=lat, tdl_longitude=lon)
                    except Exception as error:
                        row['failure_reason'] = str(error)
                    finally:
                        tas.cache_data.pop(key, None)
                else:
                    row['failure_reason'] = entry.get('failure_reason', 'Input unavailable')
                rows.append(row)
            target.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(target, index=False)
            run['streams'].append({'session': session, 'phone': phone, 'epochs': len(rows),
                                   'predictions_sha256': sha256(target)})
            save_json(out / 'prediction_manifest.json', run)
            print(f'predicted {session}/{phone}: {len(rows)} epochs', flush=True)
    if any(not torch.equal(v, before[k]) for k, v in net.state_dict().items()):
        raise RuntimeError('Model state changed')
    run.update(complete=True, model_state_unchanged=True)
    save_json(out / 'prediction_manifest.json', run)


def match_reference(gt, times, tolerance=.25):
    """Nearest UTC reference, left on exact ties; no interpolation or time fitting."""
    gt_times = gt.observation_utc_seconds.to_numpy(dtype=float)
    if not len(gt_times) or not np.isfinite(gt_times).all() or np.any(np.diff(gt_times) < 0):
        raise ValueError('Reference timestamps must be finite, nonempty and nondecreasing')
    index = np.searchsorted(gt_times, times)
    right = index.clip(0, len(gt_times)-1)
    left = (index-1).clip(0, len(gt_times)-1)
    nearest = np.where(np.abs(gt_times[right]-times) < np.abs(gt_times[left]-times), right, left)
    rows = gt.iloc[nearest].reset_index(drop=True)
    eligible = rows.gt_valid.astype(str).str.lower().eq('true').to_numpy()
    eligible &= np.isfinite(rows[['gt_latitude', 'gt_longitude']]).all(axis=1).to_numpy()
    eligible &= np.abs(gt_times[nearest]-times) <= tolerance
    return rows, eligible


def horizontal_enu(latitude, longitude, origin=(22.3, 114.18)):
    """WGS84 horizontal projection, zero height; matches the local experiment scorer."""
    def ecef(lat, lon):
        lat, lon = np.radians(lat), np.radians(lon)
        radius = 6378137. / np.sqrt(1 - 6.6943799901413165e-3 * np.sin(lat)**2)
        return np.stack([radius*np.cos(lat)*np.cos(lon), radius*np.cos(lat)*np.sin(lon),
                         radius*(1-6.6943799901413165e-3)*np.sin(lat)], axis=-1)
    lat0, lon0 = np.radians(origin)
    matrix = np.array([[-np.sin(lon0), np.cos(lon0), 0],
                       [-np.sin(lat0)*np.cos(lon0), -np.sin(lat0)*np.sin(lon0), np.cos(lat0)]])
    return (ecef(latitude, longitude)-ecef(*origin)) @ matrix.T


def metrics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {'epochs': len(values), 'median_m': float(np.median(values)) if len(values) else None,
            'p95_m': float(np.percentile(values, 95)) if len(values) else None,
            'rmse_m': float(np.sqrt(np.mean(values**2))) if len(values) else None}


def score(dataset, output):
    """Reads references only after ALL predictions have been frozen and checksummed."""
    root, out = Path(dataset), Path(output)
    run = json.loads((out / 'prediction_manifest.json').read_text())
    if not run['complete']:
        raise ValueError('All predictions must be complete before opening references')
    summaries, common_metrics, frames = [], [], []
    for record in run['streams']:
        session, phone = record['session'], record['phone']
        folder = out / 'test' / session / phone
        path = folder / 'predictions.csv'
        if sha256(path) != record['predictions_sha256']:
            raise ValueError(f'Prediction checksum mismatch: {path}')
        d = pd.read_csv(path)
        gt_path = root / 'recordings' / session / 'reference' / phone / 'trajectory-gt.csv'
        gt, eligible = match_reference(pd.read_csv(gt_path), d.time_utc_s.to_numpy())
        d['epoch_id'], d['reference_eligible'] = gt.epoch_id.to_numpy(), eligible
        truth = horizontal_enu(gt.gt_latitude.to_numpy(), gt.gt_longitude.to_numpy())
        common = eligible & d.wls_available.to_numpy() & d.tdl_available.to_numpy()
        per_common = []
        for method in ['wls', 'tdl']:
            xy = horizontal_enu(d[f'{method}_latitude'].to_numpy(), d[f'{method}_longitude'].to_numpy())
            error = np.linalg.norm(xy-truth, axis=1)
            used = eligible & d[f'{method}_available'].to_numpy() & np.isfinite(error)
            error[~used] = np.nan
            d[f'{method}_horizontal_error_m'] = error
            stat = metrics(error)
            summaries.append({'session': session, 'participant': phone, 'method': method,
                              'eligible_epochs': int(eligible.sum()), 'scored_epochs': stat.pop('epochs'),
                              'availability': float(used.sum()/eligible.sum()) if eligible.any() else None, **stat})
            stat = metrics(error[common])
            per_common.append({'method': method, 'common_epochs': stat.pop('epochs'), **stat})
            common_metrics.append({'session': session, 'participant': phone, **per_common[-1]})
        d.to_csv(folder / 'scored.csv', index=False)
        pd.DataFrame(per_common).to_csv(folder / 'common_epoch_metrics.csv', index=False)
        frames.append(d)
    pd.DataFrame(summaries).to_csv(out / 'summary.csv', index=False)
    pd.DataFrame(common_metrics).to_csv(out / 'common_epoch_metrics.csv', index=False)
    d = pd.concat(frames, ignore_index=True)
    eligible = d.reference_eligible
    common = eligible & d.wls_available & d.tdl_available
    aggregates = []
    for population, mask in [('all_available', eligible), ('common_epochs', common)]:
        for method in ['wls', 'tdl']:
            aggregates.append({'population': population, 'method': method,
                               **metrics(d.loc[mask, f'{method}_horizontal_error_m']),
                               'availability': float((eligible & d[f'{method}_available']).sum()/eligible.sum())
                               if eligible.any() else None})
    pd.DataFrame(aggregates).to_csv(out / 'aggregate.csv', index=False)
    failures = d.loc[eligible & ~d.tdl_available, 'failure_reason'].fillna('unspecified').value_counts().to_dict()
    save_json(out / 'status.json', {'complete': True, 'recordings': len(frames), 'training': False,
                                  'reference_eligible_epochs': int(eligible.sum()),
                                  'tdl_failure_reasons': {k: int(v) for k, v in failures.items()}})
    print(pd.DataFrame(aggregates).to_string(index=False))
