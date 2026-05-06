# QC_test_GHZ — Quick start for HPC vendor benchmarking

GHZ statevector benchmark with audit-grade output, built on
[QuEST](https://github.com/QuEST-Kit/QuEST) (pinned to tag `v4.2.0`).
Vendor-neutral: the program itself does not invoke any vendor-specific
tooling. QuEST is **not** bundled — `build.sh` clones it on first run.

---

## TL;DR (3 commands)

```bash
./build.sh <mode>                                    # 1. compile
./QC_test_GHZ qasm/ghz_26.qasm  out_smoke            # 2. local smoke test
mpirun -np <N> ./QC_test_GHZ qasm/ghz_38.qasm out_38 # 3. official benchmark
```

`<mode>` is one of `cpu`, `cpu-mpi`, `gpu-cuda`, `gpu-hip`. See § 1 below.

After step 3, the deliverable for review is **the entire `out_38/` directory**.

---

## 1. Build  (`./build.sh <mode>`)

The script builds QuEST and `QC_test_GHZ` together. It auto-detects the
host compiler runtime; override via env vars if needed.

| `mode` | Used for | Notes |
| --- | --- | --- |
| `cpu`      | laptop / single-node smoke test | OpenMP only, no MPI |
| `cpu-mpi`  | **CPU spec** (≤ 3072 cores, multi-node) | `mpicc` required |
| `gpu-cuda` | **GPU spec — NVIDIA** (≤ 48 cards, multi-node) | `mpicc` + CUDA toolkit |
| `gpu-hip`  | **GPU spec — AMD** (≤ 48 cards, multi-node) | `mpicc` + ROCm/HIP |

```bash
# example
CUDA_ARCH=90 ./build.sh gpu-cuda
HIP_ARCH=gfx940 ./build.sh gpu-hip
```

Switching `mode` rebuilds QuEST from scratch automatically. Re-running with
the same mode only rebuilds `QC_test_GHZ.c` (incremental).

If your linker complains about `__cxa_*` symbols, set:
```bash
CXX_RUNTIME=-lstdc++ ./build.sh cpu-mpi
```

---

## 2. Run

The binary takes exactly two arguments:

```
./QC_test_GHZ <input.qasm> <output_dir>
```

`<output_dir>` is created if missing. **Every rank writes into it**; only
rank 0 writes `log.txt`, `MANIFEST.sha256`, and `evidence/`.

### 2a. Smoke test  (1 rank, ~1 GiB RAM, ~13 s)

```bash
./QC_test_GHZ qasm/ghz_26.qasm out_smoke
```

Expected stdout:
```
output  : out_smoke
ranks   : 1
backend : CPU+OMP        # or GPU, depending on build mode
max err : 0.000e+00
result  : PASS
```

### 2b. Official 38-qubit run

The 38-qubit statevector is `2^38 × 16 B = 4 TiB`, so it must be split
across enough MPI ranks that each rank's slice fits in node memory or GPU
HBM.

> **QuEST distribution requires `<N>` to be a power of two.**

| Spec | Suggested layout | Per-rank slice |
| --- | --- | --- |
| CPU ≤ 3072 cores | `mpirun -np 32`, `OMP_NUM_THREADS=96` (32 × 96 = 3072) | 4 TiB / 32 = 128 GiB RAM |
| GPU ≤ 48 cards   | `mpirun -np 32` (1 GPU per rank, 32 of 48) | 4 TiB / 32 = 128 GiB HBM ⇢ pick rank count to fit your card |

```bash
# CPU example
export OMP_NUM_THREADS=96
mpirun -np 32 ./QC_test_GHZ qasm/ghz_38.qasm out_38

# GPU example (e.g. 1 GPU per rank)
mpirun -np 32 ./QC_test_GHZ qasm/ghz_38.qasm out_38

# SLURM equivalent (CPU)
srun --nodes=32 --ntasks-per-node=1 --cpus-per-task=96 \
     ./QC_test_GHZ qasm/ghz_38.qasm out_38
```

If a single GPU cannot hold `4 TiB / N`, double `<N>`.

---

## 3. Validate output

After the run completes:

```bash
cd out_38

# 1. headline
grep result log.txt                   # → "result = PASS"

# 2. recompute every hash
sha256sum -c MANIFEST.sha256          # → all "OK"

# 3. peek the vector (rank 0 owns global index 0)
gzcat statevector.rank0000.txt.gz | head -5
```

`log.txt` ends with:
```
acceptance
  max |amp - ref|     = 0.000e+00   at index 0
  tolerance           = 1e-08
  result              = PASS
```

---

## 4. What the audit package contains

```
out_38/
├── log.txt                       ← timings, hardware, deployment, acceptance
├── statevector.rank0000.txt.gz   ← rank 0's slice (global_index, Re, Im)
├── statevector.rank0001.txt.gz
├── ...
├── MANIFEST.sha256               ← SHA256 over everything below
└── evidence/
    ├── QC_test_GHZ.c             ← exact source compiled
    ├── QC_test_GHZ               ← exact binary that ran
    ├── ghz_38.qasm               ← input circuit (unmodified)
    └── hosts.txt                 ← rank → hostname mapping (multi-rank only)
```

**Vendor must additionally attach** the output of their own preferred
hardware-listing tool alongside this directory. The harness deliberately
does not invoke any vendor-specific tool. Examples (any one or more):

```bash
lscpu                       > out_38/evidence/lscpu.txt
nvidia-smi -L               > out_38/evidence/gpus.txt    # NVIDIA
rocm-smi --showproductname  > out_38/evidence/gpus.txt    # AMD
```

---

## 5. Spec-compliance checklist

| Requirement | How it is satisfied |
| --- | --- |
| Cross-node ≥ 2 nodes, 38 qubits | `mpirun -np <N>` with `<N>` ≥ 2; `MPI hosts` line in `log.txt` |
| GPU ≤ 48 cards / CPU ≤ 3072 cores | controlled by rank count + `OMP_NUM_THREADS` |
| QASM unmodified, no transpiler | parser supports only literal `h` / `cx` / `measure`; non-supported lines abort |
| Double precision (complex128) | `sizeof(qcomp) != 16` aborts at start; `precision` line in `log.txt` |
| Output as complex vector (Re, Im) | every rank dumps full local slice in text form |
| Max error vs reference ≤ 1e-8 | computed inline, MPI-reduced; `acceptance` block in `log.txt` |
| Auditable program + parameters | `evidence/{source, binary, qasm, hosts.txt}` + `MANIFEST.sha256` |

---

## 6. Recommended bring-up order

Do **not** jump straight to step 4 — break problems by ascending in scale.

```bash
# Step 1  toolchain check (single rank, no MPI)
./build.sh cpu
./QC_test_GHZ qasm/ghz_26.qasm  test1
# → expect: result = PASS

# Step 2  MPI build check (4 ranks, single node)
./build.sh cpu-mpi
mpirun -np 4 ./QC_test_GHZ qasm/ghz_26.qasm  test2
# → expect: result = PASS, ranks = 4, hosts = 1

# Step 3  Cross-node check (small, but spans nodes)
mpirun -np 4 -hostfile hosts4 ./QC_test_GHZ qasm/ghz_26.qasm  test3
# → expect: result = PASS, ranks = 4, hosts >= 2

# Step 4  Production GPU (or CPU) build for the official run
./build.sh gpu-cuda                  # or gpu-hip / cpu-mpi
mpirun -np 32 ./QC_test_GHZ qasm/ghz_38.qasm  out_38
```

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Linker error `__cxa_*` | C++ runtime not picked up | `CXX_RUNTIME=-lstdc++ ./build.sh ...` |
| `FATAL: qcomp != 16B` | QuEST built with `FLOAT_PRECISION=1` (single) | Re-run `./build.sh <mode>` (default is double) |
| Single rank for ≥ 32q | Non-MPI build can't fit in memory | Use `cpu-mpi` / `gpu-*` mode and `mpirun -np <N>` |
| `numNodes` != requested | QuEST distribution refused (likely non-power-of-2) | Use `<N>` ∈ {2, 4, 8, 16, 32, 64, …} |
| GPU OOM on 38q | per-GPU slice too large | Double `<N>` (halve per-GPU memory footprint) |
