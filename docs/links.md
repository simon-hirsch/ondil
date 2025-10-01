# Link functions 

## Link functions 

A link function \(g(x)\) is a smooth, monotonic function of \(x\). 

For all link functions, we implement 

- the link \(g(x)\)
- the inverse \(g^{-1}(x)\)
- the derivative of the link function $\frac{\partial g(x)}{\partial x}$.
- the first derivative _of the inverse_ of the link function \(\frac{\partial g(x)^{-1}}{\partial x}\). The choice of the inverse is justified by Equation (7) in Hirsch, Berrisch & Ziel ([2024](https://github.com/simon-hirsch/ondil/blob/main/paper.pdf)). 

The link functions implemented in `ondil` implemenent these as class methods each. Currently, we have implemented the identity-link, log-link and  shifted log-link functions.


## Overview of Link Functions

| Link Function               | Description                                                                            |
| --------------------------- | -------------------------------------------------------------------------------------- |
| `Identity`                  | Implements the identity link function \(g(x) = x\).                                    |
| `Log`                       | Implements the logarithmic link function \(g(x) = \log(x)\).                           |
| `LogShiftValue`             | Log link function with a shift value added to the inverse transformation.              |
| `LogShiftTwo`               | Log link function ensuring \(\hat{\theta} > 2\).                                       |
| `LogIdent`                  | Combines identity and log transformations.                                             |
| `Logit`                     | Implements the logit link function \(g(x) = \log(x/(1-x))\).                           |
| `Sqrt`                      | Implements the square root link function \(g(x) = \sqrt{x}\).                          |
| `SqrtShiftValue`            | Square root link function with a shift value added to the inverse transformation.      |
| `SqrtShiftTwo`              | Square root link function ensuring \(\hat{\theta} > 2\).                               |
| `InverseSoftPlus`           | Implements the inverse softplus link function.                                         |
| `InverseSoftPlusShiftValue` | Inverse softplus link function with a shift value added to the inverse transformation. |
| `InverseSoftPlusShiftTwo`   | Inverse softplus link function ensuring \(\hat{\theta} > 2\).                          |

## Shifted Link Functions

Some link functions implement _shifted_ versions. The shifted link function is implemented in the sense that the shift is added to the inverse transformation. This way, we can ensure that distribution parameters can be modelled on the continuous space of the $\eta = g(\theta)$, but in the inverse transform fullfill certain additional constraints. A common example is the $t$ distribution, where we can use a `LogShift2` or `SqrtShift2Link` to ensure that $\hat{\theta} = g^-1(\eta) > 2$ and the variance exists.

## Base Class

::: ondil.base.LinkFunction

## API Reference

::: ondil.links.Identity

### Log-Link Functions

::: ondil.links.Log

::: ondil.links.LogShiftValue

::: ondil.links.LogShiftTwo

::: ondil.links.LogIdent

### Logit Link Functions

::: ondil.links.Logit

### Square Root Link Functions

::: ondil.links.Sqrt

::: ondil.links.SqrtShiftValue

::: ondil.links.SqrtShiftTwo


### Inverse SoftPlus Link Functions

::: ondil.links.InverseSoftPlus

::: ondil.links.InverseSoftPlusShiftValue

::: ondil.links.InverseSoftPlusShiftTwo