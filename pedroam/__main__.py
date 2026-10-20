"""Usage: python -m pedroam --help."""
import argparse
import os
from pathlib import Path
import urllib.request


def main():
    # Set before importing NumPy/Torch for reproducible CPU execution.
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    from . import pipeline as p
    parser = argparse.ArgumentParser(description='Frozen official HybridShareNet on PedROAM; no training')
    sub = parser.add_subparsers(dest='command', required=True)
    fetch = sub.add_parser('fetch-weights', help='Download and checksum the pinned official legacy checkpoint')
    fetch.add_argument('--output', type=Path, required=True)
    for command in ['prepare', 'predict', 'score', 'all']:
        part = sub.add_parser(command)
        part.add_argument('--output', type=Path, required=True, help='Run directory (outside dataset)')
        if command != 'predict':
            part.add_argument('--dataset', type=Path, required=True, help='PedROAM root containing recordings/')
        if command in ['predict', 'all']:
            part.add_argument('--weights', type=Path, required=True)
        if command in ['prepare', 'all']:
            part.add_argument('--recording', action='append', help='Optional SESSION/PHONE; default ALL streams')
            part.add_argument('--limit-epochs', type=int, help='Smoke test only; default every epoch')
    args = parser.parse_args()
    if getattr(args, 'limit_epochs', None) is not None and args.limit_epochs < 1:
        parser.error('--limit-epochs must be positive')
    if args.command == 'fetch-weights':
        if args.output.exists():
            if p.sha256(args.output) != p.WEIGHT_SHA256:
                raise ValueError('Existing weight file has an unexpected checksum; not overwriting')
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            data = urllib.request.urlopen(p.WEIGHT_URL, timeout=60).read()
            import hashlib
            if hashlib.sha256(data).hexdigest() != p.WEIGHT_SHA256:
                raise ValueError('Downloaded checkpoint checksum mismatch')
            args.output.write_bytes(data)
        print(f'Official legacy checkpoint verified: {args.output}')
        return
    if args.command == 'all' and args.output.exists() and any(args.output.iterdir()):
        parser.error('all requires a new/empty output directory; existing results will not be overwritten')
    if args.command in ['prepare', 'all']:
        # Check the weight identity before expensive preprocessing in all mode.
        if args.command == 'all':
            p.load_network(args.weights)
        p.prepare(args.dataset, args.output, args.recording, args.limit_epochs)
    if args.command in ['predict', 'all']:
        p.predict(args.output, args.weights)
    if args.command in ['score', 'all']:
        p.score(args.dataset, args.output)


if __name__ == '__main__':
    main()
