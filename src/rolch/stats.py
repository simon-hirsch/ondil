from typing import Dict

import numpy as np

from rolch.gram import init_forget_vector


class IncrementalWeightedMean:

    def __init__(self, forget: float, axis: int = 0):
        self.n_observations = 0
        self.forget = forget
        self.axis = axis
        self.m = 0
        self.w = 0

    @property
    def avg(self) -> np.ndarray:
        return self.m

    @property
    def is_fitted(self) -> bool:
        return self.n_observations > 0

    def fit(self, X, sample_weights) -> None:
        self.n_observations = X.shape[0]
        forget_vector = init_forget_vector(self.forget, self.n_observations)
        self.m = np.average(X, axis=self.axis, weights=sample_weights * forget_vector)
        self.w = np.sum(sample_weights * forget_vector, axis=self.axis)

    def update_return(self, X, sample_weights) -> np.ndarray:
        self.n_observations += X.shape[0]
        nom = (self.m * self.w) * (1 - self.forget) + X * sample_weights
        den = (self.w) * (1 - self.forget) + 1
        m = nom / den

    def update_save(self, X, sample_weights) -> None:
        self.n_observations += X.shape[0]
        nom = (self.m * self.w) * (1 - self.forget) + X * sample_weights
        den = (self.w) * (1 - self.forget) + sample_weights
        self.m = nom / den
        self.w = (self.w) * (1 - self.forget) + sample_weights

    def to_dict(self) -> Dict:
        out = {
            "forget": self.forget,
            "axis": self.axis,
            "m": self.m,
            "w": self.w,
            "obs": self.n_observations,
        }
        return out

    @classmethod
    def from_dict(cls, data: dict):
        instance = cls(forget=data["forget"], axis=data["axis"])
        instance.m = np.array(data["m"]) if isinstance(data["m"], list) else data["m"]
        instance.w = np.array(data["w"]) if isinstance(data["w"], list) else data["w"]
        instance.n_observations = data["obs"]
        return instance
