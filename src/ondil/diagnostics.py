from typing import Tuple

import numpy as np
import scipy.stats as stats

from .base import DiagnosticDisply
from .error import check_matplotlib

check_matplotlib()

import matplotlib.pyplot as plt  # noqa


class PITHistogramDisplay(DiagnosticDisply):

    def __init__(self, X, y, unif):
        self.X_ = X
        self.y_ = y
        self.unif_ = unif

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        y,
        ax=None,
        figsize: Tuple[float, float] = (10, 5),
        **kwargs,
    ) -> "PITHistogramDisplay":
        predictions = estimator.predict_distribution_parameters(X)
        unif = estimator.distribution.cdf(y, predictions)
        return cls(X, y, unif).plot(ax=ax, figsize=figsize, **kwargs)

    def plot(
        self,
        ax=None,
        figsize: Tuple[float, float] = (10, 5),
        **kwargs,
    ) -> "PITHistogramDisplay":
        bins = kwargs.pop(
            "bins", np.linspace(0, 1, min(round(np.sqrt(self.y_.shape[0])) + 1, 50))
        )
        density = kwargs.pop("density", True)
        color = kwargs.pop("color", "grey")
        edgecolor = kwargs.pop("edgecolor", "black")
        lw = kwargs.pop("lw", 0.5)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.set_title("PIT Histogram")
        ax.hist(
            self.unif_,
            bins=bins,
            density=density,
            color=color,
            edgecolor=edgecolor,
            lw=lw,
        )
        ax.set_xlabel("Uniform Space")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        ax.axhline(1, color="red")
        ax.grid()

        self.ax_ = ax
        self.figure_ = ax.figure

        return self


class QQDisplay(DiagnosticDisply):
    def __init__(self, X, y, theoretical, empirical):
        self.X_ = X
        self.y_ = y
        self.theoretical_ = theoretical
        self.empirical_ = empirical

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        y,
        ax=None,
        figsize: Tuple[float, float] = (10, 5),
        **kwargs,
    ) -> "QQDisplay":
        pred = estimator.predict_distribution_parameters(X)
        quantiles = estimator.distribution.cdf(y, pred)
        quantiles = np.clip(quantiles, 1e-6, 1 - 1e-6)
        n = len(y)
        theoretical = np.linspace(1 / (n + 1), n / (n + 1), n)
        empirical = np.sort(quantiles)
        return cls(X, y, theoretical, empirical).plot(ax=ax, figsize=figsize, **kwargs)

    def plot(
        self,
        ax=None,
        figsize: Tuple[float, float] = (10, 5),
        **kwargs,
    ) -> "QQDisplay":
        color = kwargs.pop("color", "blue")
        s = kwargs.pop("s", 20)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.scatter(self.theoretical_, self.empirical_, color=color, s=s, **kwargs)
        ax.plot([0, 1], [0, 1], color="red", lw=1)
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Empirical Quantiles")
        ax.set_title("QQ Plot")
        ax.grid()
        self.ax_ = ax
        self.figure_ = ax.figure
        return self


class WormPlotDisplay(DiagnosticDisply):
    def __init__(self, X, y, xx, yy, z, lower_bound, upper_bound):
        self.X_ = X
        self.y_ = y
        self.xx_ = xx
        self.yy_ = yy
        self.z_ = z
        self.lower_bound_ = lower_bound
        self.upper_bound_ = upper_bound

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        y,
        ax=None,
        figsize: Tuple[float, float] = (10, 5),
        level: float = 0.95,
        **kwargs,
    ) -> "WormPlotDisplay":
        pred = estimator.predict(X)
        residuals = (y - pred) / np.std(y - pred)

        xx, yy = stats.probplot(residuals, fit=False)
        yy = yy - xx
        n = len(xx)
        z = np.linspace(np.min(xx), np.max(xx), n)
        p = stats.norm(loc=0, scale=1).cdf(z)
        se = (1 / stats.norm().pdf(z)) * (np.sqrt(p * (1 - p) / n))
        lower_bound = se * stats.norm.ppf((1 - level) / 2)
        upper_bound = se * stats.norm.ppf((1 + level) / 2)

        return cls(X, y, xx, yy, z, lower_bound, upper_bound).plot(
            ax=ax, figsize=figsize, **kwargs
        )

    def plot(
        self,
        ax=None,
        figsize: Tuple[float, float] = (10, 5),
        **kwargs,
    ) -> "WormPlotDisplay":
        color = kwargs.pop("color", "blue")
        alpha = kwargs.pop("alpha", 0.2)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.scatter(self.xx_, self.yy_, color=color, **kwargs)
        ax.plot(self.z_, self.lower_bound_, color="red", label="Lower confidence bound")
        ax.plot(self.z_, self.upper_bound_, color="red", label="Upper confidence bound")
        ax.fill_between(
            self.z_, self.lower_bound_, self.upper_bound_, color="grey", alpha=alpha
        )
        ax.axhline(0, color="black", lw=1, ls="--")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Empirical - Theoretical Quantiles")
        ax.set_title("Worm Plot (De-trended QQ Plot)")
        ax.grid()
        self.ax_ = ax
        self.figure_ = ax.figure
        return self


__ALL__ = ["PITHistogramDisplay", "QQDisplay", "WormPlotDisplay"]
