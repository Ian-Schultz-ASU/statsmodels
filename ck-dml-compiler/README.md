# dmlc: a compiler for Double ML inference queries

A small MLIR (Multi-Level Intermediate Representation) compiler for DML Causal inference. It takes a batch of questions asked of a fitted DML (Double Machine Learning) model, removes shared and
wasted work, and returns the same query results as [EconML](https://github.com/py-why/EconML).

It covers **inference** on a fitted EconMl's `LinearDML`: per-row effects, per-row confidence intervals, and average effects, across many treatment scenarios and many models. Fitting stays in EconML.

**Out of a 100 millisecond EconML call, about 5 are arithmetic. The rest is rebuilt for every scenario, and that is the redundancy this compiler aims to improve**

read `CompilerTLDR.md` for more info on how it all works

## Run it

Needs only Docker Desktop. The first command builds the image (a few minutes, once).

```bash
./dev make test      # build dml-opt, run 8 FileCheck tests and 207 pytest cases
./dev make results   # benchmark, profile, figures, saves results to /results 
./dev bash           # a shell in the container, repo mounted at /work
```

read `Docker_run.MD` for more info

## Example of how a query flows through

An analyst asks for the effect of three treatment changes on the same rows:

```mlir
%e0 = "dml.effect"(%X, %beta) {delta = 0.5 : f64} : (...) -> tensor<?xf64>
%e1 = "dml.effect"(%X, %beta) {delta = 1.0 : f64} : (...) -> tensor<?xf64>
%e2 = "dml.effect"(%X, %beta) {delta = 2.0 : f64} : (...) -> tensor<?xf64>
```

`--dml-decompose` splits each query into "effect size per row" and "scale it", and the stock
`--cse` pass merges the three identical effect-size ops into one:

```mlir
%0 = "dml.cate"(%X, %beta)            // computed once
%1 = "dml.scale"(%0) {a = 0.5 : f64}
%2 = "dml.scale"(%0) {a = 1.0 : f64}
%3 = "dml.scale"(%0) {a = 2.0 : f64}
```

`--dml-lower-to-linalg` turns that into standard `linalg` ops, and IREE (Intermediate
Representation Execution Environment) compiles those to machine code. Full files for each
stage are in `examples/`.

For average effects the compiler also applies **mean pushdown**: `mean(X @ b) == mean(X) @ b`,
so 25 models need one pass over the data instead of 25.

## What the measurements say

Results in `results/*.csv`, figures in `results/figures/`. Summary in short: 

- 20x to 30x faster than calling EconML once per scenario, at 20 or more scenarios.
- Mean pushdown cuts the compute for 25 models' average effects by about 9x.
- Hand-written NumPy is a strong baseline: end to end it beats the compiled code on plain
  per-row effects, because copying data into the runtime costs more than the arithmetic.

## Layout

| Path | What it holds |
|---|---|
| `dmlc/` | The C++ MLIR project: dialect, verifiers, passes, `dml-opt`, FileCheck tests |
| `python/dmlc/` | NumPy reference, MLIR emitter, pipeline runner |
| `python/tests/` | Reference vs EconML, compiled vs reference |
| `bench/` | Benchmark, profiler, figures, report generator |
| `docker/` | `Dockerfile.dev` and `compose.yaml` for the toolchain container |

## Double ML resources

1. https://docs.doubleml.org/stable/guide/basics.html
2. [Double/Debiased Machine Learning for Treatment and Causal Parameters](https://arxiv.org/abs/1608.00060)
3. [Causal Inference for the Brave and True, chapter 22](https://matheusfacure.github.io/python-causality-handbook/22-Debiased-Orthogonal-Machine-Learning.html)
