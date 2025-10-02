# Distributions

This serves as reference for all distribution objects that we implement in the `ondil` package. 

!!! note 
    This page is somewhat under construction, since `MkDocs` [does not support docstring inheritance at the moment](https://github.com/mkdocstrings/mkdocstrings/issues/78).

All distributions are based on `scipy.stats` distributions. We implement the probability density function (PDF), the cumulative density function (CDF), the percentage point or quantile function (PPF) and the random variates (RVS) accordingly as pass-through. The link functions are implemented in the same way as in GAMLSS ([Rigby & Stasinopoulos, 2005](https://academic.oup.com/jrsssc/article-abstract/54/3/507/7113027)). The link functions and their derivatives derive from the `LinkFunction` base class.


## Base Classes

| Base Distribution                          | Description                                                 |
| ------------------------------------------ | ----------------------------------------------------------- |
| [`Distribution`](#ondil.base.Distribution) | Base class for all distributions.                           |
| [`ScipyMixin`](#ondil.base.ScipyMixin)     | Base class for all distributions that are based on `scipy`. |


## List of Distributions

| Distribution                                                          | Description                            | `scipy` Base            |
| --------------------------------------------------------------------- | -------------------------------------- | ----------------------- |
| [`Normal`](#ondil.distributions.Normal)                               | Gaussian (mean and standard deviation) | `scipy.stats.norm`      |
| [`NormalMeanVariance`](#ondil.distributions.NormalMeanVariance)       | Gaussian (mean and variance)           | `scipy.stats.norm`      |
| [`StudentT`](#ondil.distributions.StudentT)                           | Student's $t$ distribution             | `scipy.stats.t`         |
| [`JSU`](#ondil.distributions.JSU)                                     | Johnson's SU distribution              | `scipy.stats.johnsonsu` |
| [`Gamma`](#ondil.distributions.Gamma)                                 | Gamma distribution                     | `scipy.stats.gamma`     |
| [`LogNormal`](#ondil.distributions.LogNormal)                         | Log-normal distribution                | `scipy.stats.lognorm`   |
| [`LogNormalMedian`](#ondil.distributions.LogNormalMedian)             | Log-normal distribution (median)       | -                       |
| [`Logistic`](#ondil.distributions.Logistic)                           | Logistic distribution                  | `scipy.stats.logistic`  |
| [`Exponential`](#ondil.distributions.Exponential)                     | Exponential distribution               | `scipy.stats.expon`     |
| [`Beta`](#ondil.distributions.Beta)                                   | Beta distribution                      | `scipy.stats.beta`      |
| [`Gumbel`](#ondil.distributions.Gumbel)                               | Gumbel distribution                    | `scipy.stats.gumbel_r`  |
| [`InverseGaussian`](#ondil.distributions.InverseGaussian)             | Inverse Gaussian distribution          | `scipy.stats.invgauss`  |
| [`BetaInflated`](#ondil.distributions.BetaInflated)                   | Beta Inflated distribution             | -                       |
| [`ReverseGumbel`](#ondil.distributions.ReverseGumbel)                 | Reverse Gumbel distribution            | `scipy.stats.gumbel_r`  |
| [`InverseGamma`](#ondil.distributions.InverseGamma)                   | Inverse Gamma distribution             | `scipy.stats.invgamma`  |
| [`BetaInflatedZero`](#ondil.distributions.BetaInflatedZero)           | Zero Inflated Beta distribution        | -                       |
| [`ZeroAdjustedGamma`](#ondil.distributions.ZeroAdjustedGamma)         | Zero Adjusted Gamma distribution       | -                       |
| [`DistributionPowerExponential`](#ondil.DistributionPowerExponential) | Power Exponential distribution         | -                       |

| Distribution                                                                                                      | Description                                              | Scale Matrix Parameterization           | Formula                                                                         |
| ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------- |
| [`MultivariateNormalInverseCholesky`](#ondil.distributions.MultivariateNormalInverseCholesky)                     | Multivariate normal (inverse Cholesky)                   | Inverse Cholesky factorization          | \$\\Sigma = (L L^{\\top})^{-1}\$, where \$L\$ is lower triangular               |
| [`MultivariateNormalInverseModifiedCholesky`](#ondil.distributions.MultivariateNormalInverseModifiedCholesky)     | Multivariate normal (inverse modified Cholesky)          | Inverse modified Cholesky factorization | \$\\Sigma = (T D T^{\\top})^{-1}\$, \$T\$ unit lower triangular, \$D\$ diagonal |
| [`MultivariateNormalInverseLowRank`](#ondil.distributions.MultivariateNormalInverseLowRank)                       | Multivariate normal (inverse low-rank)                   | Inverse low-rank factorization          | \$\\Sigma = (U U^{\\top} + D)^{-1}\$, \$U\$ low-rank, \$D\$ diagonal            |
| [`MultivariateStudentTInverseCholesky`](#ondil.distributions.MultivariateStudentTInverseCholesky)                 | Multivariate Student's \$t\$ (inverse Cholesky)          | Inverse Cholesky factorization          | \$\\Sigma = (L L^{\\top})^{-1}\$, where \$L\$ is lower triangular               |
| [`MultivariateStudentTInverseModifiedCholesky`](#ondil.distributions.MultivariateStudentTInverseModifiedCholesky) | Multivariate Student's \$t\$ (inverse modified Cholesky) | Inverse modified Cholesky factorization | \$\\Sigma = (T D T^{\\top})^{-1}\$, \$T\$ unit lower triangular, \$D\$ diagonal |
| [`MultivariateStudentTInverseLowRank`](#ondil.distributions.MultivariateStudentTInverseLowRank)                   | Multivariate Student's \$t\$ (inverse low-rank)          | Inverse low-rank factorization          | \$\\Sigma = (U U^{\\top} + D)^{-1}\$, \$U\$ low-rank, \$D\$ diagonal            |



## API Reference

::: ondil.distributions.Normal

::: ondil.distributions.NormalMeanVariance

::: ondil.distributions.StudentT

::: ondil.distributions.JSU

::: ondil.distributions.Gamma

::: ondil.distributions.LogNormal

::: ondil.distributions.LogNormalMedian

::: ondil.distributions.Logistic

::: ondil.distributions.Exponential

::: ondil.distributions.InverseGaussian

::: ondil.distributions.Beta

::: ondil.distributions.Gumbel

::: ondil.distributions.BetaInflated

::: ondil.distributions.ReverseGumbel

::: ondil.distributions.InverseGamma

::: ondil.distributions.ZeroAdjustedGamma

::: ondil.distributions.BetaInflatedZero

## Multivariate Distributions

::: ondil.distributions.MultivariateNormalInverseCholesky

::: ondil.distributions.MultivariateNormalInverseModifiedCholesky

::: ondil.distributions.MultivariateNormalInverseLowRank

::: ondil.distributions.MultivariateStudentTInverseCholesky

::: ondil.distributions.MultivariateStudentTInverseModifiedCholesky

::: ondil.distributions.MultivariateStudentTInverseLowRank

::: ondil.DistributionPowerExponential


## Base Class

::: ondil.base.Distribution

::: ondil.base.ScipyMixin
