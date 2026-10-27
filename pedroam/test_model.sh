#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${TDL_PYTHON:-${repo_root}/.venv-pedroam/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    echo "Python environment not found: $python_bin" >&2
    echo "Follow pedroam/README.md to install it, or set TDL_PYTHON to a compatible Python executable." >&2
    exit 1
fi
cd -- "$repo_root"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
exec "$python_bin" -u -m pedroam.evaluate_trained "$@"
