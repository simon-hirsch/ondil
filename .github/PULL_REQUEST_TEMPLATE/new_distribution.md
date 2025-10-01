**Please fill out the following details to help us understand your PR better and remove the unnecessary boilerplate.**
---

## New Feature: Feature Name

### Description

This PR implements the following new feature: **FILL FEATURE NAME HERE**.

This feature is designed to **FILL FEATURE DESCRIPTION HERE**.

### Checklist

- [ ] Implement the feature in the appropriate module
- [ ] Added the feature in the documentation. For more complex features, add a new section in the docs.

## Fix: Fix description

### Description

This PR fixes the following issue: **FILL ISSUE DESCRIPTION HERE**.

This fix addresses the problem by **FILL FIX DESCRIPTION HERE**.

### Checklist

- [ ] Identify the root cause of the issue
- [ ] Implement the fix in the appropriate module
- [ ] Add tests to ensure the issue is resolved

## New Distribution (replace title with the name of the distribution)

### Description

This PR introduces the **FILL DISTRIBUTION HERE**

**Please briefly describe the implementation and any reparameterizations.**

### Checklist

- [ ] Implement distribution in `src/ondil/distributions/new_dist.py`
- [ ] Add the distribution to the documentation under `docs/distributions.md`
- [ ] Add a proper docstring describing the definition of the distribution and possible parameterizations
- [ ] Implement tests for the derivatives using `rpy2` - see `tests/distributions`
- [ ] Implement a small test for the coefficients on the MTCARS test data set - see `tests/distributions`
- [ ] All tests are passing
