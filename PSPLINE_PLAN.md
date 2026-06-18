# Implementation Plan: Online P-Spline Terms

Add univariate penalized B-spline (P-spline) terms to the terms-based
`OnlineStructuredAdditiveDistributionRegressor`, by (1) building a B-spline basis +
difference-penalty module, (2) extending the numba coordinate descent to support general
quadratic penalty matrices alongside L1, and (3) adding a `PSplineTerm` that selects λ over a
grid via EDF-based information criteria — reusing the existing Gram-matrix, forgetting,
path/warm-start, and immutable-term-state machinery.

The codebase is well-prepared: terms already follow a backfitting protocol on working-vector
residuals, so each P-spline term solves its own small K×K penalized WLS subproblem.

> Status: delegated to the GitHub Copilot coding agent —
> [PR #197](https://github.com/simon-hirsch/ondil/pull/197) (base: `smooth-terms`).

## Background

The online weighted least-squares subproblem in ondil's IRLS loop is

$$
\min_\beta \tfrac12 (z - X\beta)^\top W (z - X\beta) + \text{regularization}.
$$

For a P-spline term, the regularization is a quadratic penalty
$\tfrac12 \lambda \, \beta^\top S \beta$ with $S = D_q^\top D_q$, where $D_q$ is the $q$-th
order difference operator on adjacent spline coefficients. In Gram form the inner objective
becomes

$$
\min_\beta \tfrac12 \beta^\top (G + \lambda S)\beta - h^\top \beta,
\qquad G = X^\top W X, \quad h = X^\top W z .
$$

So P-splines are not a new model class for ondil — they are a banded quadratic penalty added
to the existing online Gramian machinery.

## Steps

### Phase 1 — Basis & penalty (new file `src/ondil/terms/splines.py`)

1. B-spline basis with equidistant knots fixed from the data range at `fit` time
   (degree `p`, default cubic), evaluated via `scipy.interpolate.BSpline.design_matrix`;
   linear extrapolation beyond boundary knots for prediction (mgcv `bs="ps"` behavior).
2. Difference penalty: $D_q = \Delta^q I_K$, $S = D_q^\top D_q$ (default $q = 2$),
   normalized (e.g. by Frobenius norm) so λ has comparable meaning across terms.
3. Identifiability: center basis columns using weighted column means stored at fit time
   (sum-to-zero style); the term carries no intercept, avoiding clashes with
   `InterceptTerm` in the backfitting loop.

### Phase 2 — Solver extension (parallel with Phase 1)

4. New numba functions in `src/ondil/coordinate_descent/cd_base.py` (or a sibling module):
   a quadratic-penalty CD that uses $G + \lambda S$ in the coordinate numerator/denominator
   while keeping soft-thresholding for any L1 part, and a path variant looping a λ-grid with
   warm starts via the existing `get_start_beta`. With $S = 0$ it must reduce exactly to the
   current `online_coordinate_descent`.
5. New `QuadraticPenaltyPath(EstimationMethod)` in `src/ondil/methods/quadratic_penalty.py`:
   `_path_based_method=True`, takes the term's penalty matrix, a geometric λ-grid
   (`lambda_n`, `lambda_eps`) or fixed `lambda_`, reuses `init_gram`/`update_gram` with
   forgetting; register a string alias in `get_estimation_method`.

### Phase 3 — `PSplineTerm` (depends on 1–5)

6. Frozen `PSplineTermState` (knots, column means, `g`, `h`, `coef_path_`, `lambda_grid`,
   `edf_`, `ic_values_`, `best_idx`, `coef_`) following the `RegularizedLinearTermState`
   pattern in `src/ondil/terms/linear.py`.
7. `PSplineTerm(Term)` with `feature`, `n_splines=20`, `degree=3`, `diff_order=2`,
   `lambda_=None` (None → grid selection), `ic="aic"`, `forget` — implementing the full
   `Term` protocol (`fit` / `update` returning new instances, `predict_in_sample_during_*`,
   `predict_out_of_sample`, `make_design_matrix_*`, `_prepare_term`). Model selection mirrors
   `_LinearPathModelSelection._fit` but with
   $\mathrm{edf}(\lambda) = \mathrm{tr}\big((G + \lambda S)^{-1} G\big)$ as `n_parameters`
   instead of nonzero counts.
8. Export from `src/ondil/terms/__init__.py` and the package root; no estimator changes
   needed since `PSplineTerm` conforms to the existing term protocol consumed by
   `src/ondil/estimators/online_struct_add_distreg.py`.

### Phase 4 — Plumbing & docs

9. Verify `InformationCriterion` (`src/ondil/information_criteria.py`) handles fractional
   (float) `n_parameters`; adjust if int-assumed. Use `calculate_effective_training_length`
   for `n_observations` under forgetting.
10. Brief docs entry (terms page) + an example script under `examples/`.

## Relevant Files

- `src/ondil/terms/splines.py` — new: basis, penalty, `PSplineTerm`, `PSplineTermState`
- `src/ondil/coordinate_descent/cd_base.py` — quadratic-penalty CD + path variant
  (reuse `soft_threshold`, `get_start_beta`, active-set + tolerance logic)
- `src/ondil/methods/quadratic_penalty.py` — new `QuadraticPenaltyPath` method;
  register in `src/ondil/methods/__init__.py`
- `src/ondil/terms/linear.py` — reference: `_LinearPathModelSelection` fit/IC-selection
  pattern and immutable state handling
- `src/ondil/gram.py` — reuse `init_gram` / `update_gram` /
  `calculate_effective_training_length`
- `src/ondil/information_criteria.py` — float-edf compatibility
- `tests/test_terms_pspline.py` — new test module

## Verification

1. Basis unit tests: partition of unity inside the domain, correct shape, exact linearity of
   extrapolation beyond boundary knots.
2. Penalty tests: $S$ annihilates polynomials up to degree $q-1$; bandedness.
3. Solver equivalence: quadratic-CD with no L1 matches `np.linalg.solve(G + λS, h)` to
   tolerance; with $S = 0$ matches the existing CD path bit-for-bit on the same inputs.
4. Limit behavior: λ→large recovers degree-$(q-1)$ polynomial fit; edf monotonically
   decreasing in λ with $q \le \mathrm{edf} \le K$.
5. Batch-vs-online equivalence: `fit(X)` ≈ `fit(X[:n0])` + `update(X[n0:])` with `forget=0`
   (Gram updates are exact).
6. Integration: fit `OnlineStructuredAdditiveDistributionRegressor` with
   `[InterceptTerm, PSplineTerm]` on simulated heteroskedastic data
   (e.g. $y \sim N(\sin x, \exp(\cdot))$), check function recovery MSE and that `update` +
   `predict` run cleanly; run `pytest tests/` for regressions.

## Decisions

- Target only the terms-based estimator; the equation-based `OnlineDistributionalRegression`
  is untouched.
- λ handling v1: fixed λ or grid + EDF-based AIC/BIC selection (per term); no REML, no
  discounted predictive-score selection.
- Solver: coordinate descent extended with quadratic penalty matrices (enables future mixed
  L1+spline designs), not a separate direct solver.
- Support drift: fixed knots from initial fit + linear extrapolation; no automatic boundary
  expansion or rebasing in v1 (users can pair with `OnlineScaler` to stabilize the domain).
  Excluded: tensor products, cyclic splines, null-space reparameterization.
- Tests are internal-consistency only, no R fixtures.

## Further Considerations

1. **Null-space overlap**: with $q = 2$ the penalty leaves a linear trend unpenalized, which
   overlaps with a `LinearTerm` on the same feature. Recommendation: document it and rely on
   centering + backfitting in v1 (Option A); alternatively add an optional tiny ridge on the
   null space (Option B) or full null/range reparameterization (Option C, deferred).
2. **EDF under backfitting** is per-term (conditional on other terms), not a joint
   mgcv-style EDF — acceptable for IC-based λ selection, worth noting in docs.
3. **Knot range safety margin**: knots from the burn-in min/max can be tight; recommend a
   small configurable padding (e.g. 5% of range) so early online updates don't immediately
   hit the extrapolation region.

## Roadmap Beyond v1

- **Release 2**: automatic boundary support expansion (append/prepend knots at spacing $h$,
  initialize new coefficients by difference-extrapolation), discounted online score-based λ
  selection, edf diagnostics, cyclic P-splines.
- **Release 3**: periodic approximate REML / GCV refresh, tensor products, shrinkage
  null-space penalty / term selection.
