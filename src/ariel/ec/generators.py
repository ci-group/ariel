""""Generators and mutations for the EC module."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.random import Generator
from pydantic_settings import BaseSettings

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ── Module-level RNG (shared across generators, mutators, crossover) ──────────
SEED: int = 42
_rng: Generator = np.random.default_rng(SEED)

type Integers = Sequence[int]
type Floats = Sequence[float]


# ── Settings ──────────────────────────────────────────────────────────────────

class _GeneratorSettings(BaseSettings):
    integers_endpoint: bool = True
    choice_replace: bool = True
    choice_shuffle: bool = False


_settings: _GeneratorSettings = _GeneratorSettings()


# ── Integer generator ─────────────────────────────────────────────────────────

class IntegersGenerator:

    @staticmethod
    def integers(
        low: int,
        high: int,
        size: int | Sequence[int] | None = 1,
        *,
        endpoint: bool | None = None,
    ) -> Integers:
        ep: bool = endpoint if endpoint is not None else _settings.integers_endpoint
        return cast(
            "Integers",
            _rng.integers(low=low, high=high, size=size, endpoint=ep).astype(int).tolist(),
        )

    @staticmethod
    def choice(
        value_set: int | Integers,
        size: int | Sequence[int] | None = 1,
        probabilities: Sequence[float] | None = None,
        axis: int = 0,
        *,
        replace: bool | None = None,
        shuffle: bool | None = None,
    ) -> Integers:
        r: bool = replace if replace is not None else _settings.choice_replace
        s: bool = shuffle if shuffle is not None else _settings.choice_shuffle
        return cast(
            "Integers",
            np.array(
                _rng.choice(
                    a=value_set,
                    size=size,
                    replace=r,
                    p=probabilities,
                    axis=axis,
                    shuffle=s,
                ),
            )
            .astype(int)
            .tolist(),
        )

    @staticmethod
    def permutation(n: int) -> list[int]:
        return cast("list[int]", _rng.permutation(n).tolist())


# ── Float generator ───────────────────────────────────────────────────────────

class FloatsGenerator:

    @staticmethod
    def uniform(
        low: float,
        high: float,
        size: int | Sequence[int] | None = 1,
    ) -> Floats:
        return cast(
            "Floats",
            _rng.uniform(low=low, high=high, size=size).tolist(),
        )

    @staticmethod
    def normal(
        mean: float = 0.0,
        std: float = 1.0,
        size: int | Sequence[int] | None = 1,
    ) -> Floats:
        return cast(
            "Floats",
            _rng.normal(loc=mean, scale=std, size=size).tolist(),
        )

    @staticmethod
    def lognormal(
        mean: float = 0.0,
        sigma: float = 1.0,
        size: int | Sequence[int] | None = 1,
    ) -> Floats:
        return cast(
            "Floats",
            _rng.lognormal(mean=mean, sigma=sigma, size=size).tolist(),
        )

    @staticmethod
    def choice(
        value_set: Floats,
        size: int | Sequence[int] | None = 1,
        probabilities: Sequence[float] | None = None,
        *,
        replace: bool = True,
    ) -> Floats:
        return cast(
            "Floats",
            np.array(
                _rng.choice(
                    a=np.array(value_set),
                    size=size,
                    replace=replace,
                    p=probabilities,
                ),
            )
            .astype(float)
            .tolist(),
        )

    @staticmethod
    def linspace(
        start: float,
        stop: float,
        num: int,
    ) -> Floats:
        return cast("Floats", np.linspace(start, stop, num).tolist())


# ── Integer mutator ───────────────────────────────────────────────────────────

class IntegerMutator:

    @staticmethod
    def random_reset(
        individual: Integers,
        low: int,
        high: int,
        mutation_probability: float,
    ) -> Integers:
        arr: NDArray[np.int_] = np.asarray(individual, dtype=np.int_)
        shape = arr.shape
        replacement = _rng.integers(low=low, high=high, size=shape, endpoint=True)
        mask: NDArray[np.bool_] = _rng.random(shape) < mutation_probability
        return cast("Integers", np.where(mask, replacement, arr).astype(int).tolist())

    @staticmethod
    def integer_creep(
        individual: Integers,
        span: int,
        mutation_probability: float,
    ) -> Integers:
        arr: NDArray[np.int_] = np.array(individual, dtype=np.int_)
        shape = arr.shape
        step = _rng.integers(low=1, high=span, size=shape, endpoint=True)
        sign: NDArray[np.int_] = _rng.choice(np.array([-1, 1]), size=shape)
        gate: NDArray[np.int_] = _rng.choice(
            np.array([1, 0]),
            size=shape,
            p=[mutation_probability, 1.0 - mutation_probability],
        )
        return cast("Integers", (arr + step * sign * gate).astype(int).tolist())

    @staticmethod
    def swap(
        individual: Integers,
        mutation_probability: float,
    ) -> Integers:
        arr: list[int] = list(individual)
        n: int = len(arr)
        for idx in range(n):
            if _rng.random() < mutation_probability:
                jdx: int = int(_rng.integers(0, n))
                arr[idx], arr[jdx] = arr[jdx], arr[idx]
        return arr

    @staticmethod
    def inversion(
        individual: Integers,
        mutation_probability: float,
    ) -> Integers:
        if _rng.random() >= mutation_probability:
            return list(individual)
        arr: list[int] = list(individual)
        n: int = len(arr)
        lo, hi = sorted(
            cast("list[int]", _rng.choice(np.arange(n + 1), size=2, replace=False).tolist()),
        )
        arr[lo:hi] = arr[lo:hi][::-1]
        return arr

    @staticmethod
    def scramble(
        individual: Integers,
        mutation_probability: float,
    ) -> Integers:
        if _rng.random() >= mutation_probability:
            return list(individual)
        arr: list[int] = list(individual)
        n: int = len(arr)
        lo, hi = sorted(
            cast("list[int]", _rng.choice(np.arange(n + 1), size=2, replace=False).tolist()),
        )
        segment: list[int] = arr[lo:hi]
        _rng.shuffle(np.array(segment))  # shuffle produces NDArray; re-assign below
        shuffled: list[int] = cast("list[int]", _rng.permutation(segment).tolist())
        arr[lo:hi] = shuffled
        return arr


# ── Float mutator ─────────────────────────────────────────────────────────────

class FloatMutator:

    @staticmethod
    def gaussian(
        individual: Floats,
        std: float,
        mutation_probability: float,
        *,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ) -> Floats:
        arr: NDArray[np.float64] = np.array(individual, dtype=np.float64)
        shape = arr.shape
        noise: NDArray[np.float64] = _rng.normal(loc=0.0, scale=std, size=shape)
        mask: NDArray[np.bool_] = _rng.random(shape) < mutation_probability
        result: NDArray[np.float64] = np.where(mask, arr + noise, arr)
        if lower_bound is not None:
            result = np.maximum(result, lower_bound)
        if upper_bound is not None:
            result = np.minimum(result, upper_bound)
        return cast("Floats", result.tolist())

    @staticmethod
    def uniform_reset(
        individual: Floats,
        low: float,
        high: float,
        mutation_probability: float,
    ) -> Floats:
        arr: NDArray[np.float64] = np.array(individual, dtype=np.float64)
        shape = arr.shape
        replacement: NDArray[np.float64] = _rng.uniform(low=low, high=high, size=shape)
        mask: NDArray[np.bool_] = _rng.random(shape) < mutation_probability
        return cast("Floats", np.where(mask, replacement, arr).tolist())

    @staticmethod
    def boundary(
        individual: Floats,
        low: float,
        high: float,
        mutation_probability: float,
    ) -> Floats:
        arr: NDArray[np.float64] = np.array(individual, dtype=np.float64)
        shape = arr.shape
        boundary_vals: NDArray[np.float64] = _rng.choice(
            np.array([low, high]), size=shape,
        ).astype(np.float64)
        mask: NDArray[np.bool_] = _rng.random(shape) < mutation_probability
        return cast("Floats", np.where(mask, boundary_vals, arr).tolist())

    @staticmethod
    def polynomial(
        individual: Floats,
        low: float,
        high: float,
        mutation_probability: float,
        distribution_index: float = 20.0,
    ) -> Floats:
        arr: NDArray[np.float64] = np.array(individual, dtype=np.float64)
        mask: NDArray[np.bool_] = _rng.random(arr.shape) < mutation_probability
        u: NDArray[np.float64] = _rng.random(arr.shape)
        eta: float = distribution_index

        delta_low: NDArray[np.float64] = (arr - low) / (high - low)
        delta_high: NDArray[np.float64] = (high - arr) / (high - low)

        delta_q = np.where(
            u <= 0.5,
            (2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta_low) ** (eta + 1.0)) ** (1.0 / (eta + 1.0)) - 1.0,
            1.0 - (2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - delta_high) ** (eta + 1.0)) ** (1.0 / (eta + 1.0)),
        )
        result: NDArray[np.float64] = np.clip(
            np.where(mask, arr + delta_q * (high - low), arr), low, high,
        )
        return cast("Floats", result.tolist())

    @staticmethod
    def swap(
        individual: Floats,
        mutation_probability: float,
    ) -> Floats:
        arr: list[float] = list(individual)
        n: int = len(arr)
        for idx in range(n):
            if _rng.random() < mutation_probability:
                jdx: int = int(_rng.integers(0, n))
                arr[idx], arr[jdx] = arr[jdx], arr[idx]
        return arr

    @staticmethod
    def inversion(
        individual: Floats,
        mutation_probability: float,
    ) -> Floats:
        if _rng.random() >= mutation_probability:
            return list(individual)
        arr: list[float] = list(individual)
        n: int = len(arr)
        lo, hi = sorted(
            cast("list[int]", _rng.choice(np.arange(n + 1), size=2, replace=False).tolist()),
        )
        arr[lo:hi] = arr[lo:hi][::-1]
        return arr
