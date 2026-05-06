# QC_test_GHZ

GHZ statevector benchmark for HPC vendor acceptance, built on
[QuEST](https://github.com/QuEST-Kit/QuEST) `v4.2.0` (auto-cloned).

## Run

```bash
./build.sh <mode>                                     # cpu | cpu-mpi | gpu-cuda | gpu-hip
./QC_test_GHZ qasm/ghz_26.qasm out_smoke              # smoke test
mpirun -np <N> ./QC_test_GHZ qasm/ghz_38.qasm out_38  # benchmark; N must be 2^k
```

## Validate

```bash
grep result out_38/log.txt        # → PASS
sha256sum -c out_38/MANIFEST.sha256
```

## Output

```
out_38/
├── log.txt                       # hardware, timing, verification, result
├── statevector.rank<RRRR>.txt.gz # one per rank, "<global_idx> Re Im"
├── MANIFEST.sha256
└── evidence/{QC_test_GHZ.c, QC_test_GHZ, ghz_38.qasm}
```

The vendor attaches their own hardware listing
(`lscpu`, `nvidia-smi -L`, `rocm-smi --showproductname`, ...)
into `out_38/evidence/` alongside the program output.

## Spec compliance

| Requirement | Provided by |
| --- | --- |
| 38 qubits, ≥ 2 nodes | `mpirun -np <N>` with `N` ≥ 2 |
| GPU ≤ 48 / CPU ≤ 3072 cores | rank count + `OMP_NUM_THREADS` |
| QASM unmodified, no transpiler | parser handles only literal `h`/`cx` |
| complex128 (double precision) | aborts at start if `sizeof(qcomp) ≠ 16 B` |
| Output as complex vector (Re, Im) | every rank dumps full local slice |
| max error vs reference ≤ 1e-8 | reduced across ranks; in `log.txt` |
| Auditable program + parameters | `evidence/` + `MANIFEST.sha256` |
