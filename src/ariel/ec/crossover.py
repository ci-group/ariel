from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .generators import _rng
from .individual import JSONIterable

type _Permutation = list[int]


def _validate_same_shape(
    arr_i: NDArray[Any],
    arr_j: NDArray[Any],
) -> None:
    if arr_i.shape != arr_j.shape:
        raise ValueError(
            f"Parents must share the same shape: {arr_i.shape!r} vs {arr_j.shape!r}"
        )


def _load(
    parent_i: JSONIterable,
    parent_j: JSONIterable,
) -> tuple[tuple[int, ...], NDArray[Any], NDArray[Any]]:
    arr_i = np.array(parent_i, dtype=float)
    arr_j = np.array(parent_j, dtype=float)
    _validate_same_shape(arr_i, arr_j)
    return arr_i.shape, arr_i.flatten().copy(), arr_j.flatten().copy()


def _pack(flat: NDArray[Any], shape: tuple[int, ...]) -> JSONIterable:
    return cast("JSONIterable", flat.reshape(shape).tolist())


class Crossover:

    # -- One-point -------------------------------------------------------------

    @staticmethod
    def one_point(
        parent_i: JSONIterable,
        parent_j: JSONIterable,
    ) -> tuple[JSONIterable, JSONIterable]:
        shape, flat_i, flat_j = _load(parent_i, parent_j)
        point: int = int(_rng.integers(1, len(flat_i)))
        c1, c2 = flat_i.copy(), flat_j.copy()
        c1[point:] = flat_j[point:]
        c2[point:] = flat_i[point:]
        return _pack(c1, shape), _pack(c2, shape)

    # -- N-point ---------------------------------------------------------------

    @staticmethod
    def n_point(
        parent_i: JSONIterable,
        parent_j: JSONIterable,
        n: int,
    ) -> tuple[JSONIterable, JSONIterable]:
        shape, flat_i, flat_j = _load(parent_i, parent_j)
        length: int = len(flat_i)

        if n < 1 or n >= length:
            raise ValueError(
                f"n must be in [1, len(parent)-1], got n={n}, length={length}"
            )

        raw: NDArray[Any] = _rng.choice(
            np.arange(1, length, dtype=np.intp), size=n, replace=False
        )
        cuts: list[int] = [0, *sorted(raw.tolist()), length]

        c1, c2 = flat_i.copy(), flat_j.copy()
        for seg in range(1, len(cuts) - 1):
            lo: int = cuts[seg]
            hi: int = cuts[seg + 1]
            if seg % 2 == 1:
                c1[lo:hi] = flat_j[lo:hi]
                c2[lo:hi] = flat_i[lo:hi]

        return _pack(c1, shape), _pack(c2, shape)

    # -- Uniform ---------------------------------------------------------------

    @staticmethod
    def uniform(
        parent_i: JSONIterable,
        parent_j: JSONIterable,
        swap_probability: float = 0.5,
    ) -> tuple[JSONIterable, JSONIterable]:
        if not 0.0 <= swap_probability <= 1.0:
            raise ValueError(f"swap_probability must be in [0, 1], got {swap_probability}")

        shape, flat_i, flat_j = _load(parent_i, parent_j)
        mask: NDArray[Any] = _rng.random(len(flat_i)) < swap_probability

        c1, c2 = flat_i.copy(), flat_j.copy()
        c1[mask] = flat_j[mask]
        c2[mask] = flat_i[mask]

        return _pack(c1, shape), _pack(c2, shape)

    # -- Order crossover (OX) — for permutation representations ---------------

    @staticmethod
    def order_crossover(
        parent_i: Sequence[int],
        parent_j: Sequence[int],
    ) -> tuple[list[int], list[int]]:
        if len(parent_i) != len(parent_j):
            raise ValueError(
                f"Parents must have equal length: {len(parent_i)} vs {len(parent_j)}"
            )

        n: int = len(parent_i)
        pi: list[int] = list(parent_i)
        pj: list[int] = list(parent_j)

        cuts: list[int] = sorted(
            cast(
                "list[int]",
                _rng.choice(
                    np.arange(n + 1, dtype=np.intp), size=2, replace=False
                ).tolist(),
            )
        )
        lo: int = cuts[0]
        hi: int = cuts[1]

        def _ox(donor: list[int], other: list[int]) -> list[int]:
            segment: set[int] = set(donor[lo:hi])
            child: list[int | None] = [None] * n
            child[lo:hi] = donor[lo:hi]
            fill_positions: list[int] = [(hi + k) % n for k in range(n - (hi - lo))]
            fill_values: list[int] = [
                gene for gene in (other[hi:] + other[:hi]) if gene not in segment
            ]
            for pos, val in zip(fill_positions, fill_values):
                child[pos] = val
            return cast("list[int]", child)

        return _ox(pi, pj), _ox(pj, pi)
