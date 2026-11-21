# DML Compiler TLDR

A quick reference/learning doc covering the causal ML background, the EconML bottleneck,and the three pipeline flows that show where the compiler saves work.

## What is causal ML and why do we need a compiler for it?

### The Confounders Problem 

Causal ML big question: "if we change treatment T, what happens to outcome Y?" For example, "if we increase a drug dose, does blood pressure drop?"

You cannot just regress Y on T. Confounders, variables that influence both T and
Y, create a fake relationship on top of the real one. A patient's weight might
lead doctors to prescribe a higher dose (affects T) and independently raise blood
pressure (affects Y). **A naive regression blames the drug for what weight is
doing**

### The DML (Double Machine Learning) solution

DML paper comes up with a technique to remove confounding without knowing its exact formula:

```
Input: Y (outcome), T (treatment), X (features), W (controls)

1. Model 1: predict T from [X, W]  ->  T_resid = T - T_predicted
   (the part of treatment that covariates cannot explain)

2. Model 2: predict Y from [X, W]  ->  Y_resid = Y - Y_predicted
   (the part of outcome that covariates cannot explain)

3. Final regression: OLS (Ordinary Least Squares) on Y_resid ~ T_resid * [1, X]
   -> beta (coefficients) + cov (uncertainty)
```

Both residuals have had confounders subtracted out, so the relationship between
them is causal. "Double" = two models, two residuals.

X (features) are columns the effect is allowed to vary with: "the drug works
differently at different ages."

W (controls) soak up confounding in the helper
models but d*o not appear in the final effect formula*.

### Three names: DML vs DoubleML vs EconML

| | DML | DoubleML (library) | EconML (library) |
|---|---|---|---|
| What it is | The statistical method | Python library by the DML paper's authors | Python library by Microsoft Research (PyWhy) |
| Output | (the method, not software) | One average effect (a single number) | One effect per row (a vector) |
| Final regression | | Y_resid on T_resid alone | Y_resid on T_resid * [1, X] |

EconML's per-row output is what this project needs. One average effect from one
model has little to share. **The value comes from batches: many rows, many
scenarios, and many models on the same data**.

### LinearDML: the two arrays that survive

EconML's LinearDML estimator makes the final stage linear in X. After fitting:

- **beta** (p,): OLS (Ordinary Least Squares) *coefficients*, intercept first.
  p = d + 1 where d = number of features.
- **cov** (p, p): *covariance* matrix, measuring uncertainty in beta.

The effect for patient i is: `theta_i = beta[0] + x_i @ beta[1:]`

After `reference.extract(est)` (`python/dmlc/reference.py` copies beta and
cov from EconML's private attributes into our `FittedLinearDML` dataclass), the
helper models, residuals, and cross-validation folds are gone. Everything the
compiler does is arithmetic on these two arrays and a matrix of scoring
features X.

### The bottleneck the compiler solves


Every time you call EconML, **it recomputes shared intermediates from scratch:**

#### Normal EconML usage (what most people do):
```
est = LinearDML(...).fit(Y, T, X=X, W=W)
  └─ est is the fitted estimator object (holds helper models, beta, cov, everything)

Query 1: est.effect(X, T0=0, T1=1)
  └─ EconML internally: builds [1,X], computes X @ beta[1:], scales by delta=1 → effects
Query 2: est.effect_interval(X, T0=0, T1=1)
  └─ EconML internally: builds [1,X] AGAIN, computes X @ beta[1:] AGAIN, also cov math → intervals
Query 3: est.ate(X, T0=0, T1=1)
  └─ EconML internally: builds [1,X] AGAIN, computes X @ beta[1:] AGAIN, averages → one number
Query 4: est.effect(X, T0=0, T1=2)
  └─ EconML internally: builds [1,X] AGAIN, computes X @ beta[1:] AGAIN, scales by delta=2 → effects
```

Each call goes through est (the EconML estimator), re-enters its Python code,
rebuilds the same matrices from scratch. Calls dont know that the others exist.

#### Code terms used below

- `X`: the scoring data, one row per patient, one column per feature
- `beta`: the fitted coefficients. `beta[0]` is the baseline effect, `beta[1:]` says how
  the effect changes with each feature
- `cov`: how uncertain each coefficient in `beta` is
- `theta`: the effect for one row (one patient), `beta[0] + x @ beta[1:]`
- `cate`: the op in our code that computes `theta` for every row at once
- `se`: standard error, how uncertain one row's `theta` is (comes from `cov`)
- `delta`: the size of the treatment change being asked about (T1 - T0)
- `ate`: the average effect over all rows

#### Our Compilers Innovation: 

**The compiler sees the whole batch at once.** It extracts beta and cov once,
emits all queries into one MLIR (Multi-Level Intermediate Representation)
program, finds shared subexpressions like `X @ beta[1:]`, eliminates the
duplicates, and compiles everything into one native binary through IREE
(Intermediate Representation Execution Environment). Handles queries efficiently and gets rid of repeated work.

```
est = data.fit_econml(...)              # python/dmlc/data.py    fits LinearDML, returns EconML estimator
m = reference.extract(est)              # python/dmlc/reference.py   copies beta, cov into FittedLinearDML
                                        # est is never used again after this line

Batch all 4 queries into one MLIR program:
  reference.effects(m, X, delta=1)             # python/dmlc/reference.py     per-row effect
  reference.effect_intervals(m, X, delta=1)    # python/dmlc/reference.py     confidence intervals
  reference.ate(m, X, delta=1)                 # python/dmlc/reference.py.   average effect
  reference.effects(m, X, delta=2)             # same function, different delta

Compiler sees the batch and optimizes:
  1. theta is needed by both effects and intervals → computed once
     (CSE, Common Subexpression Elimination)
  2. theta and se do not depend on delta → each extra scenario is a cheap multiply
  3. ate uses mean pushdown: average the data first, then apply beta.
     This reads X once for all models, so it pays off with many models.
     With a single model it is extra work, not a saving.
→ one optimized program → IREE compiles and runs it → all 4 answers at once
```

## More details

Here we show 3 different situations and their pipeline flows to really help demonstrate how the compiler works and the repeated computation problem it tries to solve.

Some notation info for the Big O stuff: 
- `n_train` = number of training rows (e.g., 200,000 in our project)
- `n` = number of scoring/query rows (patients you want predictions for)
- `d` = number of features in X (e.g., 10)
- `p` = d + 1 (beta length, includes intercept)
- `K` = number of scenarios (different delta values)
- `M` = number of models
- `cv` = number of cross-validation folds (5 in our project)


#### Flow 1: First time with a NEW dataset

Here since we have the first run with a new data set we have to firs run fit the model on the new dataset, this is the bottle-neck. 

```
Step                                Who does it                          Cost                  Bottleneck?
-------------------------------------------------------------------------------------------------------------
1. Generate/load Y, T, X, W        data.make_data()                     O(n_train * d)        no
                                    (python/dmlc/data.py)

2. Fit two helper models            data.fit_econml()                    O(n_train * d * cv)   YES, slowest
   (5-fold CV, two regressions)     (python/dmlc/data.py)                cv=5, two models

3. Final OLS on residuals           EconML internally                    O(n_train * p^2)      no (p is small)

4. Copy beta, cov out of EconML     reference.extract(est)               O(p)                  no (just copy)
                     
----- EconML is done, never called again ---
5. Emit all queries as MLIR         emit.emit_effects/emit_ates()        O(M * K)              no (string ops)
                                    
6. dml-opt passes:
   a. decompose                     DMLPasses.cpp                        O(M * K) ops          no
   b. CSE merges duplicates         MLIR built-in pass                   O(M * K) ops          no
   c. lower to linalg               DMLPasses.cpp                        O(M * K) ops          no

7. IREE compile to native           pipeline.compile_module()            one-time cost         moderate
                                    (python/dmlc/pipeline.py)
8. Runner executes                  pipeline.Runner()                    see below             WHERE WE SHINE
                                    (python/dmlc/pipeline.py)
```

**What Runner executes (step 8), our compiler vs EconML:**

```
EconML (K queries, one est.effect() call each):
  K times: build [1,X] = O(n*p), matmul X @ beta = O(n*d), scale = O(n)
  Total: O(K * n * d)
  Each call re-enters Python, re-allocates [1,X], recomputes X @ beta[1:]

Our compiler (all K queries in one compiled program):
  X @ beta[1:] computed ONCE               = O(n * d)     <- EFFICIENCY  (CSE)
  Intercept added without building [1,X]   = O(n)         <- EFFICIENCY 
  Scale by each delta                      = O(K * n)
  Total: O(n*d + K*n)

For ATE (Average Treatment Effect) with M models:
  EconML: M * K times mean(X @ b_m)        = O(M * K * n * d)
  Compiler with mean pushdown:
    mean(X) computed ONCE                   = O(n * d)     <- EFFICIENCY 
    Then M*K dot products mean(X) @ b_m     = O(M * K * d) <- EFFICIENCY 
  Total: O(n*d + M*K*d) vs O(M*K*n*d)
  The n drops out of the inner loop.
```

#### Flow 2: Subsequent queries on the same dataset (same X, same model)

```
Step                                Who does it                          Cost              Bottleneck?
-------------------------------------------------------------------------------------------------------------
Steps 1-4: SKIP                     already done                         0                 no
1. Emit new queries                 emit.emit_effects/emit_ates()        O(M * K)          no
                                    (python/dmlc/emit.py)

2. dml-opt passes                   DMLPasses.cpp                        O(M * K)          no

3. IREE compile                     pipeline.compile_module()            one-time          moderate
                                    (python/dmlc/pipeline.py)

4. Runner executes                  pipeline.Runner()                    O(n*d + K*n)      fast
                                    (python/dmlc/pipeline.py)

vs EconML which STILL pays O(K * n * d) per batch because it has
no memory that X @ beta[1:] was computed before.
```

A new batch of queries needs a recompile (steps 5-7), because the scenarios and
the number of models are fixed inside the compiled program. **Compiling costs more
than a single run, so the payoff comes from reusing one compild program across new
data or a refitted model** (Flow 3)

#### Flow 3: Dataset changes (new patients, or model is refit)

```
Case A: New scoring rows X_new (same fitted model)
--------------------------------------------------
  - beta and cov stay the same, no refitting
  - Re-run steps 5-8 with X_new
  - Or just: pipeline.Runner(module)(X_new, beta)   <- skip even recompilation
    because n is dynamic (the MLIR module uses ? for the row dimension)
  - Cost: O(n_new * d) for the execution

Case B: Model is refit (new training data)
------------------------------------------
  - Must redo steps 1-4: refit via data.fit_econml() = O(n_train * d * cv)   <- BOTTLENECK
  - reference.extract(est) copies new beta, cov
  - Steps 5-7 can reuse the same compiled module if d is unchanged
    (the MLIR module is parametric in beta values, they are runtime inputs)
  - pipeline.Runner(module)(X, new_beta)   <- no recompilation needed
  - Cost: fitting is slow, but inference stays fast

Case C: Feature count d changes
-------------------------------
  - Must refit via data.fit_econml() AND recompile via pipeline.compile_module(),
    because d is baked into the MLIR module as a static dimension
  - This is the most expensive case: full pipeline from scratch
```

### Cost comparison summary

| Scenario | EconML cost per query batch | Compiler cost | Savings |
|---|---|---|---|
| K effects, 1 model | O(K * n * d) | O(n * d + K * n) | K-1 redundant matmuls eliminated |
| M * K ATEs | O(M * K * n * d) | O(n * d + M * K * d) | n drops out of inner loop |
| Repeat query, same X | O(K * n * d) again | Same cost again. Results are not cached between runs, but nothing is recompiled | No refitting, no recompiling |
| New X, same model | O(K * n * d) | O(n * d + K * n), no recompile | Dynamic n, module reusable |

The compiler's advantage grows with K (more scenarios), M (more models), and n
(more rows for ATE (Average Treatment Effect) pushdown). The fitting step is
always EconML's job and always the slowest part, but it happens once.
