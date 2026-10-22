# Repository Guidelines

## Project Structure & Module Organization

- Root scripts implement the main workflow: `preprocess.py`, `train.py`, `evaluate.py`, and `baseline.py`.
- `model/model.py` defines PyTorch networks and normalization layers. Root training uses the nine-feature `HybridShareSysNet`; PedROAM uses the pretrained three-feature `HybridShareNet`.
- `config/` contains JSON dataset and run configurations. `image/` holds README diagrams.
- `pedroam/` provides dataset discovery, preprocessing, frozen inference, scoring, and a module CLI. Consult `pedroam/README.md` for setup and `pedroam/VALIDATION.md` for validation limits.
- `tests/test_pedroam.py` contains automated integration and numerical-behavior tests.

## Build, Test, and Development Commands

Run commands from the repository root. There is no separate build step. For the tested PedROAM environment, use Python 3.10 in a virtual environment:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r pedroam/requirements.txt
```

Configure local dataset paths before using the root workflow:

```bash
python preprocess.py config/train.json       # Cache training observations
python train.py config/train.json            # Train the model
python preprocess.py config/test_klt1.json   # Cache evaluation observations
python evaluate.py config/test_klt1.json /path/to/model.pth
python baseline.py config/test_klt1.json      # Evaluate the baseline
```

For PedROAM, run `python -m pedroam --help`. Download verified weights with `python -m pedroam fetch-weights --output checkpoints/hybrid_share.pth`; use the documented `all` command with a fresh output directory.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions and variables, `PascalCase` classes, and uppercase constants. Follow surrounding Python style and keep changes focused; no formatter or linter configuration is currently checked in. Preserve feature order, units, array shapes, and checkpoint compatibility when changing numerical code.

## Testing Guidelines

Run `python -m unittest discover -s tests -v`. Name test files `test_*.py` and methods `test_*`; use `unittest.TestCase` and temporary directories for fixtures. Enable checkpoint validation with `TDL_TEST_WEIGHTS=checkpoints/hybrid_share.pth` before the test command. Add regression cases for changed filtering, matching, or scoring behavior. No coverage threshold is configured.

## Commit & Pull Request Guidelines

Recent commits use short, action-oriented subjects such as “Add frozen pretrained PedROAM preprocessing and evaluation entrypoint”; no mandatory prefix is established. Describe the change, affected workflow, validation commands/results, and relevant issues in PRs. Include metric comparisons when positioning behavior changes.

## Data & Reproducibility

Keep datasets, caches, environments, downloaded checkpoints, and generated results untracked. Load only trusted pickle caches. Preserve PedROAM’s separation of GNSS-only preprocessing, frozen inference, and reference-based scoring.

## Data & Outputs

Treat `/home/maxson/data/PedROAM` as input data; inspect write paths before running experiments. Save temporary results under repository-local `outputs/` and official results under `results/<session-name>/<pedestrian-number>/<test-name>/`. Keep datasets, checkpoints, and generated outputs out of commits.