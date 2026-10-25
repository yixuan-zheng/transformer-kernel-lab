# Reproduce the measurements

Core dependencies are pinned in pyproject.toml. Use the minimal CUDA wheel setup in the README. The measured environment uses PyTorch 2.4.1+cu121, Triton 3.0.0 and NVIDIA L40S. Wheel installation, reference tests, CLI entry points and installed FP16/BF16 operators passed validation in an isolated environment inheriting those CUDA dependencies. A clean download and installation of the entire CUDA stack has not been tested.

## Run on an allocated L40S

```bash
python -m unittest discover -s tests -v
python -m tests.validate_kernels
tkl-bench --output results/new-run
```

Use a fresh output directory. The benchmark checks correctness and warms each implementation before timing. Seed 2026 determines input generation and randomized ordering. Rows are 1/8/32/128/512/2048; both FP16 and BF16 are measured. Normalization widths are 1024/2048/4096/8192. SwiGLU adds 11008/14336/28672 as primary intermediate dimensions. Five interleaved batches preserve median, IQR and range. Graph samples target 15 ms; event samples average 30 invocations. A bounded sweep uses four FP16 development shapes and three samples per launch candidate.

## Capture memory counters separately

With Nsight Compute available, on the allocated GPU:

```bash
mkdir -p results/new-counters
ncu --target-processes all --kernel-name 'regex:_add|_norm' \
  --launch-count 3 --csv \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum \
  --log-file results/new-counters/nsight-metrics.csv \
  python -m benchmarks.study --output results/new-counters --ncu
```

Inspect kernel IDs/names before aggregating. The capture covers separate addition and normalization, then fused residual normalization at [2048,8192] FP16. Replay/cache behavior differs from the timing loop; delayed writeback can make per-kernel DRAM writes incomplete. These counters are not a benchmark-loop bandwidth measurement.

## Analyze the saved study on CPU

The included plots used Python 3.11, Matplotlib 3.11.2 and NumPy 2.4.6. On that Python version:

```bash
python -m pip install -r results/l40s/plot-requirements.txt
cp -R results/l40s results/analysis-copy
python benchmarks/analyze.py results/analysis-copy
python benchmarks/report.py results/analysis-copy
```

Compatible plotting dependencies for Python 3.9 can instead be installed with `pip install '.[plots]'`. Analysis regenerates CSV summaries, four plots and audited counts from the twelve files in `traces/`. Only events categorized as `kernel` count as launches; GPU annotation markers are excluded. Report generation expects this saved matrix and the Nsight CSV; it is not a general report generator for arbitrary matrices. Use a copy to preserve the saved evidence.

## Verify the published files

```bash
python tools/verify_publication.py
```

PUBLICATION.sha256 covers every published file except the manifest itself. The CPU workflow runs reference tests, builds distributions and checks the manifest. It does not run GPU performance tests.
