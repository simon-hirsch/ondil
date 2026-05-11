# CLAUDE.md — orientation for Claude on the `ondil` package

Short pointer file for future sessions. Read this first; use it to
locate things before grepping.

## Ground rules (Project II)

- **Do not change anything without explicit permission from Christian.**
  Discussion, planning, and reading are always fine. Edits require
  a green light.
- The repo lives inside a git submodule; `.git` points at
  `../.git/modules/ondil`. Claude's sandboxed bash cannot run `git`
  commands directly — Christian runs them in his terminal.
- When asked to change code, prefer the smallest surgical diff;
  gate new behavior behind a conditional rather than replacing
  existing behavior outright.

## Package layout (relevant bits)

```
src/ondil/
  base/
    estimation_method.py       # EstimationMethod ABC; compute_edf lives here
    distribution.py            # Distribution, CopulaMixin, BivariateCopulaMixin
  methods/
    elasticnet.py              # ElasticNetPath; compute_edf impl; _last_lambda_path cache
    lasso_path.py              # Just ElasticNetPath(alpha=1.0)
    ridge.py                   # Single-lambda ridge; NOT path-based -> no model selection
    recursive_least_squares.py # OLS, non-path; bypasses _fit_model_selection
    factory.py                 # get_estimation_method("ols"|"lasso"|"elasticnet")
  estimators/
    online_mvdistreg.py        # ~2200 lines; the main estimator class
    online_gamlss.py           # univariate cousin
  distributions/
    bicop_normal.py, bicop_clayton.py, bicop_gumbel.py, bicop_studentt.py
    mv_normal_*.py, mv_t_*.py  # multivariate (cholesky / low-rank variants)
  coordinate_descent.py        # numba-jit coord descent; penalty convention below
  gram.py                      # init_gram / update_gram (weighted, with forget)
  information_criteria.py      # InformationCriterion(n_observations, n_parameters).from_ll
  links/copulalinks.py         # FisherZLink, KendallsTauToParameter, etc.
examples/
  edf_bivariate_copula_demo.py # working bivariate copula EDF demo
  dv_bivariate_fit.py          # Christian's exploratory script
```

## Key conventions

- **Bold notation in the paper**: `\vbeta`, `\mX`, `\mW`, `\veta`, `\vz`,
  `\vu` — use these macros when editing `paper_online_copula/paper/main.tex`.
- **Coordinate descent penalty** (`coordinate_descent.py`): L2 diagonal
  addition is `lambda * w_j * (1 - alpha)` (weights linear, NOT squared);
  L1 soft-threshold amount is `alpha * lambda * w_j`.
- **`_regularization_allowed = {p: bool}`** on distributions means AD-R
  structural regularization, NOT elastic-net regularization. Bivariate
  copulas all have `{0: False}` — that's fine, elastic net still works.
- **`is_regularized_[p][k]`** (on the estimator) is the per-coefficient
  boolean mask for the coord descent. Intercept defaults to `False`
  (not regularized) when `regularize_intercept=False`.

## The EDF work (done)

Three files were modified, gated on `CopulaMixin` with scalar dependence:

- `base/estimation_method.py`: added `compute_edf(x_gram, beta_path, is_regularized)`
  with `NotImplementedError` default.
- `methods/elasticnet.py`:
  - `fit_beta_path` / `update_beta_path` now cache `self._last_lambda_path`.
  - `compute_edf` implements `trace[G_A (G_A + lambda * J_A)^{-1}]` with
    `J_A = diag((1 - alpha) * w_j)` on the active set; zero on
    unregularized columns. LassoPath inherits this for free.
- `estimators/online_mvdistreg.py`:
  - `_fit_model_selection` around L1399–L1425 and
    `_update_model_selection` around L1505–L1530 both gate on
    `issubclass(distribution.__class__, CopulaMixin) and n_params == 1 and n_dist_elements_[param] == 1`.
  - Inside the gate: `n_parameters = method.compute_edf(...)`.
  - Otherwise: unchanged non-zero count.

Sanity checks that must keep holding:
1. `alpha=1` (LASSO) => `compute_edf == |active set|` exactly.
2. `alpha=0` (Ridge) => EDF monotone in lambda, bounded by rank.
3. `alpha=0.5` (EN) => `EDF <= |A|` per lambda.

The demo script `examples/edf_bivariate_copula_demo.py` prints the
pure-LASSO identity as its final check.

## Navigation cheatsheet for `online_mvdistreg.py`

- `class MultivariateOnlineDistributionalRegressionPath` starts ~L126.
- Path-based model-selection call site: ~L1201 inside `_fit`.
- `_fit_model_selection`: ~L1349 (fit phase).
- `_update_model_selection`: ~L1421 (online update phase).
- `count_nonzero_coef` / `count_coef_to_be_fitted`: ~L860, ~L877.
- `CopulaMixin` gates scattered: L982, L1148, L1226, L1545, L1594, L1671, L1810, L2052.
- Early stopping uses its own nonzero count at ~L810, independent of
  the model-selection branch — do not conflate.
- `_x_gram[p][k][a]` is the IRLS-weighted X^T W X with forget factor
  already applied; shape `(adr_steps, n_features, n_features)`.

## What Claude does NOT know yet (flag if relevant)

- The outer/inner iteration state machine and convergence logic in full.
- AD-R regularization internals (`_adr_distance`, `_adr_mapping_index_to_max_distance`).
- Multivariate distribution implementations (only the interface).
- How `overshoot_correction`, `dampen_estimation`, `weight_delta`
  participate in updates.
- Scaler / sklearn-compat plumbing.

## Things that burned context in past sessions

- Reading the whole `online_mvdistreg.py` instead of grepping to a
  line number. Prefer `Grep` + targeted `Read` with `offset`/`limit`.
- Trying to install scipy / tlmgr packages in the sandbox
  (blocked — don't retry).
- Running pdflatex in the sandbox — fails on `commath.sty` missing.
  Christian compiles locally.
- Long exploratory responses when Christian has already decided.

## Working style preferences (from Christian)

- Direct and honest over hedged. Push back when warranted.
- Keep prose tight. Only expand when asked.
- When implementing a discussed plan, implement — don't re-justify.
- Explicit permission is required before editing files.
