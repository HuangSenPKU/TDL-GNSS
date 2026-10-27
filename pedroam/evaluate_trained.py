"""Evaluate the fixed 0421-AM first-80% checkpoint without training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from . import training as t

np, pd, torch, p = t.np, t.pd, t.torch, t.p
REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_RUN = REPOSITORY / 'results/0421-AM-2026/all/tdl-gnss-scratch-80-20-20261027'


def select_streams(streams, training_session, cutoff, scope):
    """Preserve the archived UTC boundary; never evaluate the training prefix."""
    selected = []
    for stream in streams:
        meta = stream['meta']
        is_training_session = meta['session'] == training_session
        if scope == 'other-sessions' and is_training_session:
            continue
        if scope == 'heldout-tail' and not is_training_session:
            continue
        current = {'meta': meta.copy(), 'entries': stream['entries']}
        if is_training_session:
            current['entries'] = [e for e in current['entries'] if e['time_utc_s'] >= cutoff]
            current['reference_time_bounds'] = (cutoff, np.inf)
            current['evaluation_set'] = 'heldout_tail'
        else:
            current['evaluation_set'] = 'other_sessions'
        if current['entries']:
            selected.append(current)
    if not selected:
        raise ValueError('No epochs match the requested scope')
    return selected


def summarize(frame, methods):
    """Direct epoch-weighted means, with separate common/available populations."""
    rows = []
    for evaluation_set, subset in frame.groupby('evaluation_set', sort=True):
        groups = [('overall', 'all', 'all', subset)]
        groups += [('session', session, 'all', group) for session, group in subset.groupby('session')]
        groups += [('phone', session, phone, group)
                   for (session, phone), group in subset.groupby(['session', 'phone'])]
        for scope, session, phone, group in groups:
            eligible = group.reference_eligible.to_numpy(dtype=bool)
            common = eligible & np.isfinite(group[methods]).all(axis=1).to_numpy()
            for population, mask in [('common_epochs', common), ('all_available', eligible)]:
                for method in methods:
                    values = group.loc[mask, method].to_numpy(dtype=float)
                    values = values[np.isfinite(values)]
                    available = eligible & np.isfinite(group[method].to_numpy(dtype=float))
                    rows.append({'evaluation_set': evaluation_set, 'scope': scope,
                        'session': session, 'phone': phone, 'population': population,
                        'method': method, 'eligible_epochs': int(eligible.sum()),
                        'availability': float(available.sum()/eligible.sum()) if eligible.any() else None,
                        'mean_m': float(values.mean()) if len(values) else None, **p.metrics(values)})
    return pd.DataFrame(rows)


def check_output_path(path, dataset):
    path, dataset = Path(path).resolve(), Path(dataset).resolve()
    # The user may explicitly choose the dataset's results/ folder; never allow
    # experiment output inside recordings/ or the other input directories.
    if path == dataset or (dataset in path.parents and dataset/'results' not in path.parents):
        raise ValueError('Inside the dataset, output is allowed only under results/')
    return path


def verify_cache_inputs(dataset, metadata, hashes):
    """Reject a cache from a different input revision, not just a corrupt pickle."""
    folder = dataset/'recordings'/metadata['session']
    observations = list((folder/'phones'/metadata['phone']/'gnsslogger').glob('*_v302.26o'))
    if len(observations) != 1:
        raise ValueError('Expected exactly one converted observation file per stream')
    expected = [(observations[0], metadata['obs_sha256'])]
    expected += [(folder/'navigation'/name, digest) for name, digest in metadata['navigation_sha256'].items()]
    actual_nav = {path.name for path in (folder/'navigation').glob('*.rnx')}
    if actual_nav != set(metadata['navigation_sha256']):
        raise ValueError(f'Navigation files changed: {folder}; use --rebuild-cache')
    for path, digest in expected:
        if path not in hashes:
            hashes[path] = p.sha256(path)
        if hashes[path] != digest:
            raise ValueError(f'Input differs from cached revision: {path}; use --rebuild-cache')


def write_report(output, summary, config):
    labels = {'wls': '等权 WLS', 'official': '官方冻结模型', 'model': '本地从头训练模型'}
    lines = ['# 固定模型测试结果', '',
             f"权重 SHA256：`{config['weights_sha256']}`。本次不训练、不重新估计归一化。", '',
             'mean 为逐历元水平误差的算术平均值，单位为米；整体值直接汇总历元计算。', '',
             '共同历元用于精度对比；可用率始终以该组全部有效参考历元为分母。', '']
    for name, title in [('heldout_tail', '0421-AM 后20%'), ('other_sessions', '其他场次')]:
        subset = summary[(summary.evaluation_set == name) & (summary.scope == 'overall')
                         & (summary.population == 'common_epochs')]
        if subset.empty:
            continue
        lines += [f'## {title}', '', '| 方法 | 历元数 | mean/m | median/m | P95/m | RMSE/m | 可用率 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in subset.itertuples():
            def number(value, percent=False):
                if pd.isna(value):
                    return '—'
                return f'{value:.2%}' if percent else f'{value:.2f}'
            lines.append(f'| {labels[row.method]} | {row.epochs} | {number(row.mean_m)} | '
                         f'{number(row.median_m)} | {number(row.p95_m)} | {number(row.rmse_m)} | '
                         f'{number(row.availability, True)} |')
        lines.append('')
    lines += ['## 文件', '', '- `summary.csv`：两类测试、各场次、各手机的完整指标。',
              '- `mean.csv`：mean 对照表；`overall.csv`：两类测试的整体指标。',
              '- `scored/`：逐历元水平误差；`predictions_model/`：本模型位置与失败原因。',
              '- `config.json`、`status.json`、`data_inventory.json`：配置、完成状态与数据校验。', '',
              '这两类测试范围不同，分开报告。既有实验曾查看过这些数据，不代表全新盲测。', '']
    (output/'REPORT.md').write_text('\n'.join(lines))


def run(args):
    dataset = args.dataset.resolve()
    training_run = args.training_run.resolve()
    training_config = json.loads((training_run/'config.json').read_text())
    selection = json.loads((training_run/'selection.json').read_text())
    if training_config.get('selection_rule') != 'fixed_epochs_final_checkpoint':
        raise ValueError('--training-run must be the chronological fixed-epoch training archive')
    training_session = training_config['session']
    cutoff = float(training_config['cutoff_utc_s'])
    weights = training_run/'final.pth'
    if p.sha256(weights) != selection['weights_sha256']:
        raise ValueError('Checkpoint differs from the archived selected model')
    pairs = p.discover(dataset)
    if args.scope == 'heldout-tail':
        pairs = [pair for pair in pairs if pair[0] == training_session]
    elif args.scope == 'other-sessions':
        pairs = [pair for pair in pairs if pair[0] != training_session]
    if not pairs:
        raise ValueError('No recordings match the requested scope')
    run_name = args.run_name or datetime.now(timezone.utc).strftime('test_%Y%m%dT%H%M%S_%fZ')
    if Path(run_name).name != run_name or run_name in ['', '.', '..']:
        raise ValueError('--run-name must be a single directory name')
    output = check_output_path(args.output_root/run_name, dataset)
    output.mkdir(parents=True, exist_ok=False)

    def log(message):
        print(message, flush=True)
        with (output/'run.log').open('a') as handle:
            handle.write(message+'\n')

    start = time.monotonic()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(output=str(output), weights=str(weights), weights_sha256=selection['weights_sha256'],
                  training=False, normalization_refitted=False, training_session=training_session,
                  cutoff_utc_s=cutoff, selected_epoch=selection['selected_epoch'],
                  versions={name: p.version(name) for name in ['torch', 'numpy', 'pandas', 'pymap3d', 'tasgnss', 'pyrtklib']})
    p.save_json(output/'config.json', config)
    p.save_json(output/'status.json', {'complete': False, 'stage': 'loading'})
    try:
        torch.set_num_threads(1)
        from model.model import HybridShareNet
        net = HybridShareNet().double()
        net.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        net.requires_grad_(False)
        original = {k: v.clone() for k, v in net.state_dict().items()}
        official_net = None
        if not args.no_official:
            official_net, _ = p.load_network(args.baseline_weights)
        source = output/'source'
        source.mkdir()
        for path in [Path(__file__), Path(t.__file__), Path(p.__file__), REPOSITORY/'model/model.py',
                     Path(t.tas.__file__).parent/'core.py']:
            shutil.copy2(path, source/path.name)
        shutil.copy2(training_run/'selection.json', output/'training_selection.json')
        shutil.copy2(training_run/'config.json', output/'training_config.json')
        cache_root = args.cache_root.resolve()
        if args.rebuild_cache or not (cache_root/'preprocess_manifest.json').exists():
            cache_root = output/'preprocessing'
            log('Preparing fresh GNSS-only caches from RINEX...')
            p.prepare(dataset, cache_root, ['/'.join(pair) for pair in pairs])
        manifest = json.loads((cache_root/'preprocess_manifest.json').read_text())
        if not manifest['complete'] or manifest['reference_used']:
            raise ValueError('Require complete GNSS-only preprocessing')
        wanted = set(pairs)
        metadata = [m for m in manifest['streams'] if (m['session'], m['phone']) in wanted]
        if {(m['session'], m['phone']) for m in metadata} != wanted:
            raise ValueError('Cache lacks selected recordings; use --rebuild-cache')
        manifest = {**manifest, 'streams': metadata}
        hashes = {}
        for item in metadata:
            verify_cache_inputs(dataset, item, hashes)
        p.save_json(output/'cache_manifest.json', manifest)
        config.update(actual_cache_root=str(cache_root), cache_manifest_sha256=p.sha256(output/'cache_manifest.json'))
        p.save_json(output/'config.json', config)
        loaded = t.load_streams(cache_root, manifest, {s for s, _ in pairs})
        streams = select_streams(loaded, training_session, cutoff, args.scope)
        log(f'Output: {output}')
        log(f'Predicting {len(streams)} streams, {sum(len(s["entries"]) for s in streams)} epochs; references not attached.')
        predicted = t.predict_streams(net, streams, output/'predictions_model')
        official = t.predict_streams(official_net, streams, output/'predictions_official') if official_net is not None else None
        if not all(torch.equal(v, net.state_dict()[k]) for k, v in original.items()):
            raise RuntimeError('Model state changed during inference')
        if p.sha256(weights) != selection['weights_sha256']:
            raise RuntimeError('Checkpoint file changed')
        log('Predictions complete; matching references and scoring.')
        inventory = t.attach_references(streams, dataset)
        p.save_json(output/'data_inventory.json', inventory)
        t.comparison(streams, official, predicted, output/'scored', 'model')
        frames = []
        for stream in streams:
            meta = stream['meta']
            frame = pd.read_csv(output/'scored'/f"{meta['session']}__{meta['phone']}.csv")
            frame['evaluation_set'] = stream['evaluation_set']
            frames.append(frame)
        methods = ['wls'] + (['official'] if official is not None else []) + ['model']
        summary = summarize(pd.concat(frames, ignore_index=True), methods)
        summary.to_csv(output/'summary.csv', index=False)
        summary[summary.scope == 'overall'].to_csv(output/'overall.csv', index=False)
        summary[summary.scope == 'session'].to_csv(output/'per_session.csv', index=False)
        summary[summary.scope == 'phone'].to_csv(output/'per_phone.csv', index=False)
        mean = summary.pivot(index=['evaluation_set', 'scope', 'session', 'phone', 'population'], columns='method', values='mean_m')
        mean.rename(columns=lambda name: name+'_mean_m').reset_index().to_csv(output/'mean.csv', index=False)
        # Replace the helper's older aggregate with the same complete metrics.
        summary.to_csv(output/'scored/summary.csv', index=False)
        write_report(output, summary, config)
        log(summary[(summary.scope == 'overall') & (summary.population == 'common_epochs')].to_string(index=False))
        p.save_json(output/'status.json', {'complete': True, 'training': False, 'model_state_unchanged': True,
            'normalization_refitted': False, 'references_used_only_after_predictions': True,
            'streams': len(streams), 'raw_test_epochs': sum(len(s['entries']) for s in streams),
            'elapsed_seconds': time.monotonic()-start})
        log('Complete. Read REPORT.md, mean.csv and summary.csv.')
        p.save_json(output/'archive_manifest.json', {str(path.relative_to(output)): p.sha256(path)
            for path in sorted(output.rglob('*')) if path.is_file()})
    except Exception as error:
        p.save_json(output/'status.json', {'complete': False, 'error': str(error)})
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('/home/maxson/data/PedROAM'))
    parser.add_argument('--training-run', type=Path, default=DEFAULT_TRAINING_RUN)
    parser.add_argument('--cache-root', type=Path, default=REPOSITORY/'outputs/pedroam_all_20261022')
    parser.add_argument('--output-root', type=Path, default=REPOSITORY/'pedroam/results')
    parser.add_argument('--run-name', help='Optional new subdirectory name; default unique UTC timestamp')
    parser.add_argument('--scope', choices=['all-unseen', 'heldout-tail', 'other-sessions'], default='all-unseen')
    parser.add_argument('--baseline-weights', type=Path, default=REPOSITORY/'checkpoints/hybrid_share.pth')
    parser.add_argument('--no-official', action='store_true', help='Compare only local model and equal-weight WLS')
    parser.add_argument('--rebuild-cache', action='store_true', help='Recompute GNSS-only caches in the new run directory')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
