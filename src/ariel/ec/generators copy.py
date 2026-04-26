""" "Generators and mutations for the EC module."""

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
    """
    Internal configuration for generator and mutator defaults.

    Parameters
    ----------
    integers_endpoint : bool, optional
        Whether the ``high`` bound is inclusive when sampling integers.
        Default is ``True``.
    choice_replace : bool, optional
        Whether sampling with replacement is the default for
        ``IntegersGenerator.choice``. Default is ``True``.
    choice_shuffle : bool, optional
        Whether the output of ``IntegersGenerator.choice`` is shuffled
        by default. Default is ``False``.
    """

    integers_endpoint: bool = True
    choice_replace: bool = True
    choice_shuffle: bool = False


_settings: _GeneratorSettings = _GeneratorSettings()


# ── Integer generator ─────────────────────────────────────────────────────────


class IntegersGenerator:
    """Namespace of static methods for generating integer sequences."""

    @staticmethod
    def integers(
        low: int,
        high: int,
        size: int | Sequence[int] | None = 1,
        *,
        endpoint: bool | None = None,
    ) -> Integers:
        """
        Draw random integers from a uniform discrete distribution.

        Parameters
        ----------
        low : int
            Lowest integer to be drawn (inclusive).
        high : int
            Upper boundary. Inclusive when ``endpoint`` is ``True``,
            exclusive otherwise.
        size : int or Sequence[int] or None, optional
            Output shape. Default is ``1``.
        endpoint : bool or None, optional
            Whether ``high`` is included in the range. Falls back to
            ``_settings.integers_endpoint`` when ``None``.

        Returns
        -------
        Integers
            A flat list of sampled integers with length ``size``.
        """
        ep: bool = (
            endpoint if endpoint is not None else _settings.integers_endpoint
        )
        return cast(
            "Integers",
            _rng.integers(
                low=low,
                high=high,
                size=size,
                endpoint=ep,
            )
            .astype(int)
            .tolist(),
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
        """
        Draw a random sample from a given integer set or range.

        Parameters
        ----------
        value_set : int or Integers
            If an ``int``, samples are drawn from ``range(value_set)``.
            Otherwise, samples are drawn from the provided sequence.
        size : int or Sequence[int] or None, optional
            Output shape. Default is ``1``.
        probabilities : Sequence[float] or None, optional
            Probabilities associated with each element of ``value_set``.
            Must sum to 1. When ``None``, a uniform distribution is used.
        axis : int, optional
            Axis along which to draw samples when ``value_set`` is
            multi-dimensional. Default is ``0``.
        replace : bool or None, optional
            Whether to sample with replacement. Falls back to
            ``_settings.choice_replace`` when ``None``.
        shuffle : bool or None, optional
            Whether to shuffle the output. Falls back to
            ``_settings.choice_shuffle`` when ``None``.

        Returns
        -------
        Integers
            A list of sampled integers.
        """
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
        """
        Return a random permutation of integers ``0`` to ``n - 1``.

        Parameters
        ----------
        n : int
            Upper bound (exclusive) of the integer range to permute.

        Returns
        -------
        list[int]
            A randomly ordered list of integers from ``0`` to ``n - 1``.
        """
        return cast("list[int]", _rng.permutation(n).tolist())


# ── Float generator ───────────────────────────────────────────────────────────


class FloatsGenerator:
    """Namespace of static methods for generating float sequences."""

    @staticmethod
    def uniform(
        low: float,
        high: float,
        size: int | Sequence[int] | None = 1,
    ) -> Floats:
        """
        Draw samples from a continuous uniform distribution.

        Parameters
        ----------
        low : float
            Lower boundary of the distribution (inclusive).
        high : float
            Upper boundary of the distribution (exclusive).
        size : int or Sequence[int] or None, optional
            Output shape. Default is ``1``.

        Returns
        -------
        Floats
            A list of uniformly sampled floats.
        """
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
        """
        Draw samples from a Gaussian (normal) distribution.

        Parameters
        ----------
        mean : float, optional
            Mean of the distribution. Default is ``0.0``.
        std : float, optional
            Standard deviation of the distribution. Default is ``1.0``.
        size : int or Sequence[int] or None, optional
            Output shape. Default is ``1``.

        Returns
        -------
        Floats
            A list of normally distributed floats.
        """
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
        """
        Draw samples from a log-normal distribution.

        Parameters
        ----------
        mean : float, optional
            Mean of the underlying normal distribution. Default is ``0.0``.
        sigma : float, optional
            Standard deviation of the underlying normal distribution.
            Default is ``1.0``.
        size : int or Sequence[int] or None, optional
            Output shape. Default is ``1``.

        Returns
        -------
        Floats
            A list of log-normally distributed floats.
        """
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
        """
        Draw a random sample from a given float sequence.

        Parameters
        ----------
        value_set : Floats
            The pool of float values to sample from.
        size : int or Sequence[int] or None, optional
            Output shape. Default is ``1``.
        probabilities : Sequence[float] or None, optional
            Probabilities associated with each element of ``value_set``.
            Must sum to 1. When ``None``, a uniform distribution is used.
        replace : bool, optional
            Whether to sample with replacement. Default is ``True``.

        Returns
        -------
        Floats
            A list of sampled floats.
        """
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
        """
        Return evenly spaced floats over a specified interval.

        Parameters
        ----------
        start : float
            The starting value of the sequence (inclusive).
        stop : float
            The end value of the sequence (inclusive).
        num : int
            Number of evenly spaced samples to generate.

        Returns
        -------
        Floats
            A list of ``num`` evenly spaced floats from ``start`` to ``stop``.
        """
        return cast("Floats", np.linspace(start, stop, num).tolist())


# ── Integer mutator ───────────────────────────────────────────────────────────


class IntegerMutator:
    """Namespace of static mutation operators for integer-encoded individuals."""

    @staticmethod
    def random_reset(
        individual: Integers,
        low: int,
        high: int,
        mutation_probability: float,
    ) -> Integers:
        """
        Replace each gene with a random integer drawn uniformly from ``[low, high]``.

        Each gene is independently mutated with probability
        ``mutation_probability``.

        Parameters
        ----------
        individual : Integers
            The integer-encoded genotype to mutate.
        low : int
            Lower bound of the replacement range (inclusive).
        high : int
            Upper bound of the replacement range (inclusive).
        mutation_probability : float
            Per-gene probability of mutation. Must be in ``[0.0, 1.0]``.

        Returns
        -------
        Integers
            A new genotype with randomly reset genes.
        """
        arr: NDArray[np.int_] = np.asarray(individual, dtype=np.int_)
        shape = arr.shape
        replacement = _rng.integers(
            low=low, high=high, size=shape, endpoint=True
        )
        mask: NDArray[np.bool_] = _rng.random(shape) < mutation_probability
        return cast(
            "Integers", np.where(mask, replacement, arr).astype(int).tolist()
        )

    @staticmethod
    def integer_creep(
        individual: Integers,
        span: int,
        mutation_probability: float,
    ) -> Integers:
        """
        Apply creep mutation by adding a small random step to each gene.

        Each gene is independently perturbed by a random integer step drawn
        from ``[1, span]`` in a random direction, with the mutation gated by
        ``mutation_probability``.

        Parameters
        ----------
        individual : Integers
            The integer-encoded genotype to mutate.
        span : int
            Maximum absolute step size for each mutation (inclusive).
        mutation_probability : float
            Per-gene probability of mutation. Must be in ``[0.0, 1.0]``.

        Returns
        -------
        Integers
            A new genotype with creep-mutated genes.
        """
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
        """
        Randomly swap pairs of genes within the individual.

        Each gene is independently considered for a swap with a randomly
        chosen other gene, with probability ``mutation_probability``.

        Parameters
        ----------
        individual : Integers
            The integer-encoded genotype to mutate.
        mutation_probability : float
            Per-gene probability of initiating a swap. Must be in
            ``[0.0, 1.0]``.

        Returns
        -------
        Integers
            A new genotype with swapped genes.
        """
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
        """
        Reverse a randomly selected sub-sequence of the genotype.

        With probability ``mutation_probability``, two cut points are chosen
        uniformly at random and the segment between them is reversed in place.

        Parameters
        ----------
        individual : Integers
            The integer-encoded genotype to mutate.
        mutation_probability : float
            Probability that inversion occurs. Must be in ``[0.0, 1.0]``.

        Returns
        -------
        Integers
            A new genotype with the selected segment reversed, or an
            unchanged copy if mutation did not occur.
        """
        if _rng.random() >= mutation_probability:
            return list(individual)
        arr: list[int] = list(individual)
        n: int = len(arr)
        lo, hi = sorted(
            cast(
                "list[int]",
                _rng.choice(np.arange(n + 1), size=2, replace=False).tolist(),
            ),
        )
        arr[lo:hi] = arr[lo:hi][::-1]
        return arr

    @staticmethod
    def scramble(
        individual: Integers,
        mutation_probability: float,
    ) -> Integers:
        """
        Randomly shuffle a sub-sequence of the genotype.

        With probability ``mutation_probability``, two cut points are chosen
        uniformly at random and the segment between them is shuffled in place.

        Parameters
        ----------
        individual : Integers
            The integer-encoded genotype to mutate.
        mutation_probability : float
            Probability that scramble occurs. Must be in ``[0.0, 1.0]``.

        Returns
        -------
        Integers
            A new genotype with the selected segment scrambled, or an
            unchanged copy if mutation did not occur.
        """
        if _rng.random() >= mutation_probability:
            return list(individual)
        arr: list[int] = list(individual)
        n: int = len(arr)
        lo, hi = sorted(
            cast(
                "list[int]",
                _rng.choice(np.arange(n + 1), size=2, replace=False).tolist(),
            ),
        )
        segment: list[int] = arr[lo:hi]
        _rng.shuffle(
            np.array(segment)
        )  # shuffle produces NDArray; re-assign below
        shuffled: list[int] = _rng.permutation(segment).tolist()
        arr[lo:hi] = shuffled
        return arr


# ── Float mutator ─────────────────────────────────────────────────────────────


class FloatMutator:
    """Namespace of static mutation operators for real-valued individuals."""

    @staticmethod
    def gaussian(
        individual: Floats,
        std: float,
        mutation_probability: float,
        *,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ) -> Floats:
        """
        Perturb each gene by adding Gaussian noise.

        Each gene is independently mutated with probability
        ``mutation_probability`` by adding zero-mean Gaussian noise with
        standard deviation ``std``. Optional bounds clip the result.

        Parameters
        ----------
        individual : Floats
            The real-valued genotype to mutate.
        std : float
            Standard deviation of the Gaussian noise.
        mutation_probability : float
            Per-gene probability of mutation. Must be in ``[0.0, 1.0]``.
        lower_bound : float or None, optional
            Minimum allowable gene value after mutation. No clipping applied
            when ``None``. Default is ``None``.
        upper_bound : float or None, optional
            Maximum allowable gene value after mutation. No clipping applied
            when ``None``. Default is ``None``.

        Returns
        -------
        Floats
            A new genotype with Gaussian-perturbed genes.
        """
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
        """
        Replace each gene with a value drawn uniformly from ``[low, high)``.

        Each gene is independently mutated with probability
        ``mutation_probability``.

        Parameters
        ----------
        individual : Floats
            The real-valued genotype to mutate.
        low : float
            Lower bound of the replacement range (inclusive).
        high : float
            Upper bound of the replacement range (exclusive).
        mutation_probability : float
            Per-gene probability of mutation. Must be in ``[0.0, 1.0]``.

        Returns
        -------
        Floats
            A new genotype with uniformly reset genes.
        """
        arr: NDArray[np.float64] = np.array(individual, dtype=np.float64)
        shape = arr.shape
        replacement: NDArray[np.float64] = _rng.uniform(
            low=low, high=high, size=shape
        )
        mask: NDArray[np.bool_] = _rng.random(shape) < mutation_probability
        return cast("Floats", np.where(mask, replacement, arr).tolist())

    @staticmethod
    def boundary(
        individual: Floats,
        low: float,
        high: float,
        mutation_probability: float,
    ) -> Floats:
        """
        Reset each gene to either the lower or upper boundary value.

        Each gene is independently mutated with probability
        ``mutation_probability`` by replacing it with either ``low`` or
        ``high``, chosen at random with equal probability.

        Parameters
        ----------
        individual : Floats
            The real-valued genotype to mutate.
        low : float
            Lower boundary value.
        high : float
            Upper boundary value.
        mutation_probability : float
            Per-gene probability of mutation. Must be in ``[0.0, 1.0]``.

        Returns
        -------
        Floats
            A new genotype with boundary-reset genes.
        """
        arr: NDArray[np.float64] = np.array(individual, dtype=np.float64)
        shape = arr.shape
        boundary_vals: NDArray[np.float64] = _rng.choice(
            np.array([low, high]),
            size=shape,
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
        """
        Apply polynomial mutation as defined in Deb & Goyal (1996).

        Each gene is independently mutated with probability
        ``mutation_probability`` using a polynomial perturbation that
        respects the ``[low, high]`` bounds. A higher ``distribution_index``
        produces perturbations closer to the parent gene.

        Parameters
        ----------
        individual : Floats
            The real-valued genotype to mutate.
        low : float
            Lower bound of the valid gene range (inclusive).
        high : float
            Upper bound of the valid gene range (inclusive).
        mutation_probability : float
            Per-gene probability of mutation. Must be in ``[0.0, 1.0]``.
        distribution_index : float, optional
            Controls the spread of the mutation distribution. Larger values
            concentrate mutations near the original gene. Default is ``20.0``.

        Returns
        -------
        Floats
            A new genotype with polynomial-mutated genes, clipped to
            ``[low, high]``.
        """
        arr: NDArray[np.float64] = np.array(individual, dtype=np.float64)
        mask: NDArray[np.bool_] = _rng.random(arr.shape) < mutation_probability
        u: NDArray[np.float64] = _rng.random(arr.shape)
        eta: float = distribution_index

        delta_low: NDArray[np.float64] = (arr - low) / (high - low)
        delta_high: NDArray[np.float64] = (high - arr) / (high - low)

        delta_q = np.where(
            u <= 0.5,
            (2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta_low) ** (eta + 1.0))
            ** (1.0 / (eta + 1.0))
            - 1.0,
            1.0
            - (
                2.0 * (1.0 - u)
                + 2.0 * (u - 0.5) * (1.0 - delta_high) ** (eta + 1.0)
            )
            ** (1.0 / (eta + 1.0)),
        )
        result: NDArray[np.float64] = np.clip(
            np.where(mask, arr + delta_q * (high - low), arr),
            low,
            high,
        )
        return cast("Floats", result.tolist())

    @staticmethod
    def swap(
        individual: Floats,
        mutation_probability: float,
    ) -> Floats:
        """
        Randomly swap pairs of genes within the individual.

        Each gene is independently considered for a swap with a randomly
        chosen other gene, with probability ``mutation_probability``.

        Parameters
        ----------
        individual : Floats
            The real-valued genotype to mutate.
        mutation_probability : float
            Per-gene probability of initiating a swap. Must be in
            ``[0.0, 1.0]``.

        Returns
        -------
        Floats
            A new genotype with swapped genes.
        """
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
        """
        Reverse a randomly selected sub-sequence of the genotype.

        With probability ``mutation_probability``, two cut points are chosen
        uniformly at random and the segment between them is reversed in place.

        Parameters
        ----------
        individual : Floats
            The real-valued genotype to mutate.
        mutation_probability : float
            Probability that inversion occurs. Must be in ``[0.0, 1.0]``.

        Returns
        -------
        Floats
            A new genotype with the selected segment reversed, or an
            unchanged copy if mutation did not occur.
        """
        if _rng.random() >= mutation_probability:
            return list(individual)
        arr: list[float] = list(individual)
        n: int = len(arr)
        lo, hi = sorted(
            cast(
                "list[int]",
                _rng.choice(
                    np.arange(n + 1),
                    size=2,
                    replace=False,
                ).tolist(),
            ),
        )
        arr[lo:hi] = arr[lo:hi][::-1]
        return arr
