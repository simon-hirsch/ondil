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

| Distribution                                                    | Description                            | `scipy` Base            |
| --------------------------------------------------------------- | -------------------------------------- | ----------------------- |
| [`Normal`](#ondil.distributions.Normal)                         | Gaussian (mean and standard deviation) | `scipy.stats.norm`      |
| [`NormalMeanVariance`](#ondil.distributions.NormalMeanVariance) | Gaussian (mean and variance)           | `scipy.stats.norm`      |
| [`StudentT`](#ondil.distributions.StudentT)                     | Student's $t$ distribution             | `scipy.stats.t`         |
| [`JSU`](#ondil.distributions.JSU)                               | Johnson's SU distribution              | `scipy.stats.johnsonsu` |
| [`Gamma`](#ondil.distributions.Gamma)                           | Gamma distribution                     | `scipy.stats.gamma`     |
| [`LogNormal`](#ondil.distributions.LogNormal)                   | Log-normal distribution                | `scipy.stats.lognorm`   |
| [`LogNormalMedian`](#ondil.distributions.LogNormalMedian)       | Log-normal distribution (median)       | -                       |
| [`Logistic`](#ondil.distributions.Logistic)                     | Logistic distribution                  | `scipy.stats.logistic`  |
| [`Exponential`](#ondil.distributions.Exponential)               | Exponential distribution               | `scipy.stats.expon`     |
| [`Beta`](#ondil.distributions.Beta)                             | Beta distribution                      | `scipy.stats.beta`      |
| [`Gumbel`](#ondil.distributions.Gumbel)                         | Gumbel distribution                    | `scipy.stats.gumbel_r`  |
| [`InverseGaussian`](#ondil.distributions.InverseGaussian)       | Inverse Gaussian distribution          | `scipy.stats.invgauss`  |
| [`BetaInflated`](#ondil.distributions.BetaInflated)             | Beta Inflated distribution             | -                       |
| [`ReverseGumbel`](#ondil.distributions.ReverseGumbel)           | Reverse Gumbel distribution            | `scipy.stats.gumbel_r`  |
| [`InverseGamma`](#ondil.distributions.InverseGamma)             | Inverse Gamma distribution             | `scipy.stats.invgamma`  |
| [`SkewT1`](#ondil.distributions.SkewT1)                         | SkewT Type 1 distribution              | -                       |
| [`BetaInflatedZero`](#ondil.distributions.BetaInflatedZero)     | Zero Inflated Beta distribution        | -                       |
| [`ZeroAdjustedGamma`](#ondil.distributions.ZeroAdjustedGamma)   | Zero Adjusted Gamma distribution       | -                       |

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

::: ondil.distributions.SkewT1

::: ondil.distributions.ZeroAdjustedGamma

::: ondil.distributions.BetaInflatedZero

## Base Class

::: ondil.base.Distribution

::: ondil.base.ScipyMixin
