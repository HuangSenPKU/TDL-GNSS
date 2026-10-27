# PedROAM supervised training

`python -m pedroam.training` is a separate experimental training entrypoint.
The existing `python -m pedroam` commands remain the frozen official benchmark.
Locally trained checkpoints are not accepted by that benchmark's pinned weight loader.

The three-feature `HybridShareNet` supports two initialization modes:

- `--initialization scratch`: PyTorch's random initialization for all learned
  layers. Population mean and standard deviation are computed only from satellite
  observations in eligible training samples, with standard deviations floored at
  `1e-6`. No checkpoint or pretrained normalization is read for initialization.
  `--weights` is forbidden in this mode.
- `--initialization finetune` (default): load the verified official checkpoint
  from `--weights`, including its fixed normalization.

Both modes preserve the feature order C/N0, elevation in radians, initial WLS
residual in metres. Every solve
restarts from the cached GNSS-only initial position and clock biases. The input
cache must have been generated locally: checksum verification does not make an
untrusted pickle safe to load.

Reference positions are matched by UTC with the existing 0.25-second tolerance.
Only valid references and eligible GNSS inputs enter training. Supervision uses
mean horizontal distance after converting predicted ECEF positions to geodetic
latitude/longitude, setting height to zero, and projecting into the same ENU
frame as the existing scorer. No reference height, attitude or original car
lever arm is used. The Torch projection is checked against the existing scorer
and with numerical gradient checking.

All recordings within a requested session belong to the same split. Splits must
be nonempty, known and disjoint. Training and validation references are loaded
before optimization; test references are read only after the chosen checkpoint
is saved and all test predictions are complete. Existing historical evaluations
have already included these sessions, so this is a retrospective held-out split,
not a new blind dataset.

## From-scratch experiment

### One-session chronological 80/20 experiment

For a strict first-80% train / last-20% test split with no validation set:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-pedroam/bin/python -m pedroam.temporal_training \
  --dataset /home/maxson/data/PedROAM \
  --cache-root outputs/pedroam_all_20261022 \
  --session 0421-AM-2026 --train-fraction 0.8 \
  --output outputs/pedroam_0421am_80_20_new_run \
  --baseline-weights checkpoints/hybrid_share.pth \
  --epochs 30 --batch-size 64 --lr 0.001 --grad-clip 10 --seed 20261027
```

This entrypoint is always from scratch. It pools all phones in the requested
session and sorts raw observation epochs by UTC before any input/reference
filtering. One shared cutoff assigns the first 80% to train and final 20% to test.
Equal timestamps are kept together on the test side if they straddle the cutoff;
this can make the training fraction slightly smaller. For 0421-AM, the split is
exactly 5,984 / 1,496 raw epochs. Independent per-phone splits would overlap in
time because the phone start times differ by about 13 seconds.

Reference rows are restricted to their own side of the cutoff before nearest
matching. Both subsets share a physical reference CSV per phone, but test-side
labels are excluded from training matching and normalization. Training samples
are shuffled only after the chronological partition is fixed. The test labels
are attached for scoring after final-checkpoint selection and predictions.

The run uses all eligible samples in the first 80%, with no extra validation
partition or early stopping. The epoch count is fixed before training, and
`final.pth` is the final epoch, not a checkpoint chosen by test error. The exact
partition is saved in `split_membership.csv`. Do not compare earlier local models
that trained on the entire 0421-AM session as held-out baselines on this test tail.

### Multiple-session experiment

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-pedroam/bin/python -m pedroam.training \
  --dataset /home/maxson/data/PedROAM \
  --cache-root outputs/pedroam_all_20261022 \
  --initialization scratch \
  --baseline-weights checkpoints/hybrid_share.pth \
  --output outputs/pedroam_scratch_new_run \
  --train-session 0421-AM-2026 \
  --train-session 0422-AM1-2026 \
  --train-session 0422-PM-2026 \
  --validation-session 0421-PM-2026 \
  --validation-session 0422-AM2-2026 \
  --test-session 0423-AM-2026 \
  --test-session 0423-PM-2026 \
  --epochs 30 --patience 6 --batch-size 64 --lr 0.001 \
  --grad-clip 10 --seed 20261027
```

`--baseline-weights` is optional and is used only for official-model comparisons
after training and checkpoint selection have finished. It is never passed to the
scratch initializer or optimizer. Omit it to run without loading any pretrained
checkpoint at any stage. Normalization is fitted once and remains fixed during
training. `initial.pth` and `initialization.json` record the exact random starting
state and training-only normalization for auditing.

This experiment deliberately allows more optimization than the earlier 8-epoch
fine-tuning trial. Their results are not a controlled comparison of initialization
alone: the learning rates and maximum training budgets differ.

## Earlier fine-tuning experiment

From the repository root, with the tested `.venv-pedroam` environment:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-pedroam/bin/python -m pedroam.training \
  --dataset /home/maxson/data/PedROAM \
  --cache-root outputs/pedroam_all_20261022 \
  --weights checkpoints/hybrid_share.pth \
  --output outputs/pedroam_finetune_new_run \
  --train-session 0421-AM-2026 \
  --train-session 0422-AM1-2026 \
  --train-session 0422-PM-2026 \
  --validation-session 0421-PM-2026 \
  --validation-session 0422-AM2-2026 \
  --test-session 0423-AM-2026 \
  --test-session 0423-PM-2026 \
  --epochs 8 --patience 3 --batch-size 64 --lr 0.0001 \
  --grad-clip 10 --seed 20261027
```

Use a new output directory. No input dataset files are written. Reusing existing
GNSS caches avoids recomputing satellite geometry but does not reuse model
predictions or test references for training. The first experiment has 15,403
training candidates across five phone streams. Every candidate is attempted once
per epoch, in seeded shuffled order; there is no stride or training subsampling.

Adam updates use the mean gradient over 64 successful epochs (or the actual
remaining count in the last batch), followed by gradient-norm clipping at 10.
Backward is performed separately per GNSS epoch to bound graph memory. There is
no 200-metre error cutoff. Solver failures are counted and skipped; nonfinite
gradients stop the run. CPU float64 is used throughout.

Checkpoint selection minimizes validation mean horizontal error, assigning a
fixed 1,000-metre penalty to each eligible reference without a successful
prediction. Successful errors are not clipped. The denominator is all eligible
validation references, including rejected GNSS inputs. Epoch zero (random for
scratch, official for fine-tuning) is a selection candidate. `--patience` controls
early stopping: six non-improving epochs in the scratch command above, three in
the earlier fine-tuning command. This selection objective is distinct from RMSE
and median error. Neither mode can select the official comparison model as a
fallback unless that model was its actual initialization.

## Outputs

- `config.json`, `data_inventory.json`, `source/`: hyperparameters, dependency
  versions, reference/cache hashes and source snapshots.
- `history.json`: training loss, solver failures, optimizer steps and validation
  metrics for each epoch.
- `initial.pth`, `initialization.json`: exact initial state and its provenance.
- `epoch_NNN.pth`, `best.pth`, `selection.json`: state dictionaries and the
  validation-selected checkpoint identity. Normalization is included.
- `validation_epoch_NNN/`, `test_official/`, `test_scratch/` or `test_finetuned/`: predictions,
  per-epoch failure reasons and checksums.
- `validation_comparison/summary.csv`, `test_comparison/summary.csv`: WLS,
  official (when requested) and learned model metrics, both individually available
  and common to all methods; aggregate and per-phone rows are included.
- `status.json`: completion status and aggregate results.

Report median, P95, RMSE and availability together, including the common-epoch
comparison. Generated artifacts remain untracked. Official experiment archives
belong under `results/<session-name>/<pedestrian-number>/<test-name>/`.

```bash
TDL_TEST_WEIGHTS=checkpoints/hybrid_share.pth \
  .venv-pedroam/bin/python -m unittest discover -s tests -v
```
