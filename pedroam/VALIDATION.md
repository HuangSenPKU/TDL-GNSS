# Integration validation

Date: 2026-10-20. Python 3.10, Linux, CPU; dependency versions in README and requirements.txt.

- Seven unittest cases passed, including the opt-in strict official-checkpoint loading test.
- `fetch-weights` downloaded the pinned official checkpoint and verified its SHA256.
- Full end-to-end CLI run on `0423-PM-2026/P02`: fresh RINEX/navigation preprocessing at all 1,555 raw epochs, frozen inference, then scoring. Predictions, scored rows and common-epoch metrics exactly matched the previously completed experiment (1,539 eligible, 1,496 TDL outputs).
- All 13-stream inference/scoring regression: reused the previously computed full-epoch GNSS-only initial-WLS caches, selecting the same three input features, without references or trained-model predictions in those caches. All 13 prediction files, scored files, per-stream common metrics and aggregate metrics exactly matched the previous full-release run. This check did not rerun raw preprocessing for all 13 streams a second time.
- Model parameters and normalization buffers unchanged after inference.
- Upstream Python sources and requirements.txt were not modified. Root readme.md only links to the new interface. Changes are confined to the dataset integration package, tests, documentation and .gitignore.
- No private dataset, reference file, runtime cache, environment, downloaded checkpoint or raw prediction file is committed.

Full-release totals: 32,505 raw epochs; 31,929 eligible reference matches; 30,005 TDL outputs. On common epochs, WLS/TDL median 41.00/24.63 m, P95 112.59/80.34 m, RMSE 58.94/56.06 m. These are offline integration results, not a claim of exact reproduction of the paper.

Not validated: installation in a newly provisioned environment, GPU/other Python versions, exact checkpoint training provenance, complete reference/timestamp audit or real-time broadcast reception availability.
