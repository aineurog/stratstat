# Benchmarks

`benchmark_numba.py` times each numba kernel against its numpy fallback and
prints a table of results.

## Run

```bash
python benchmarks/benchmark_numba.py
```

Run from the repository root. The script needs numpy. It detects numba at
runtime and skips the numba columns when numba is not installed.

## Output

Each row reports:

* `compile (ms)`: the first call to the numba kernel, which includes the one
  time JIT compile cost.
* `numba (ms)`: the per call time after compilation.
* `numpy (ms)`: the numpy fallback per call time.
* `speedup`: `numpy / numba`. Values above 1.0 mean numba is faster.

The compile column explains why the library gates numba behind
`numba_worthwhile` in `core/_utils.py`. For small workloads the numpy fallback
finishes before numba finishes compiling, so small inputs stay on the numpy
path.
