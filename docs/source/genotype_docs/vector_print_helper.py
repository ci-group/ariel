from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Iterable, Optional
import numpy as np

from rich.console import Console
from rich.text import Text
from rich.traceback import install

install()
console = Console()
log = console.log

rng = np.random.default_rng(42)


# ---------------------------
# Pretty printing helpers
# ---------------------------

def rich_vector(
    vec: Sequence[float],
    styles: Optional[Sequence[str]] = None,
    bracket_style: str = "bold",
    sep_style: str = "bold",
) -> Text:
    """Render a vector with optional per-element styles."""
    t = Text("[", style=bracket_style)
    for i, v in enumerate(vec):
        if i > 0:
            t.append(", ", style=sep_style)

        style = styles[i] if styles is not None else ""
        # Print ints without .0, floats nicely
        if isinstance(v, (int, np.integer)) or (isinstance(v, float) and float(v).is_integer()):
            s = str(int(v))
        else:
            s = f"{float(v):.3f}"
        t.append(s, style=style)
    t.append("]", style=bracket_style)
    return t


def styles_by_segments(n: int, cut_points: Sequence[int], left_style: str, right_style: str) -> List[str]:
    """
    Create per-element styles with a single 'boundary' for two segments.
    For more complex segmenting, use segment_styles_from_sources below.
    """
    # Not used directly; kept for clarity / future extensions
    return [left_style] * n


def segment_styles_from_sources(
    n: int,
    sources: Sequence[int],               # 0 means from A, 1 means from B (or more generally: parent index)
    palette: Sequence[str],
) -> List[str]:
    """Map 'sources' (per-gene parent origin) to rich styles via palette."""
    return [palette[sources[i] % len(palette)] for i in range(n)]


# ---------------------------
# Crossover operators
# ---------------------------

@dataclass
class CrossResult:
    child: List[float]
    sources: List[int]  # per-gene origin: 0 from parent A, 1 from parent B
    info: str           # human-readable summary


def one_point_crossover(a: Sequence[float], b: Sequence[float], point: int) -> CrossResult:
    """
    One-point crossover: child = a[:point] + b[point:].
    point is an index in [1, len(a)-1].
    """
    n = len(a)
    if n != len(b):
        raise ValueError("Parents must have the same length.")
    if not (1 <= point <= n - 1):
        raise ValueError(f"point must be in [1, {n-1}]")

    child = list(a[:point]) + list(b[point:])
    sources = [0] * point + [1] * (n - point)
    return CrossResult(child=child, sources=sources, info=f"one-point @ {point}")


def n_point_crossover(a: Sequence[float], b: Sequence[float], points: Sequence[int]) -> CrossResult:
    """
    N-point crossover: alternate segments between A and B using sorted cut points.
    Example points=[3,7] => segments: [0:3] from A, [3:7] from B, [7:n] from A.
    """
    n = len(a)
    if n != len(b):
        raise ValueError("Parents must have the same length.")
    pts = sorted(set(points))
    if any(p <= 0 or p >= n for p in pts):
        raise ValueError(f"All points must be in [1, {n-1}] and unique.")

    cuts = [0] + pts + [n]
    child: List[float] = []
    sources: List[int] = []
    take_from = 0  # start with A

    for i in range(len(cuts) - 1):
        start, end = cuts[i], cuts[i + 1]
        segment = a[start:end] if take_from == 0 else b[start:end]
        child.extend(segment)
        sources.extend([take_from] * (end - start))
        take_from = 1 - take_from

    return CrossResult(child=child, sources=sources, info=f"n-point @ {pts}")


def uniform_crossover(a: Sequence[float], b: Sequence[float], p_from_a: float = 0.5) -> CrossResult:
    """
    Uniform crossover: for each gene choose from A with prob p_from_a else from B.
    """
    n = len(a)
    if n != len(b):
        raise ValueError("Parents must have the same length.")
    if not (0.0 <= p_from_a <= 1.0):
        raise ValueError("p_from_a must be in [0,1].")

    mask = rng.random(n) < p_from_a  # True -> take from A
    child = [a[i] if mask[i] else b[i] for i in range(n)]
    sources = [0 if mask[i] else 1 for i in range(n)]
    return CrossResult(child=child, sources=sources, info=f"uniform p(A)={p_from_a:.2f}")


# ---------------------------
# Mutation operators
# ---------------------------

@dataclass
class MutResult:
    mutated: List[float]
    changed: List[bool]  # per-gene whether it changed (for coloring)
    info: str


def swap_mutation(x: Sequence[float], i: int, j: int) -> MutResult:
    """Swap two positions i and j."""
    n = len(x)
    if not (0 <= i < n and 0 <= j < n):
        raise ValueError("swap indices out of range.")
    y = list(x)
    y[i], y[j] = y[j], y[i]
    changed = [k in (i, j) and (i != j) for k in range(n)]
    return MutResult(mutated=y, changed=changed, info=f"swap ({i}<->{j})")


def gaussian_mutation(
    x: Sequence[float],
    sigma: float = 0.5,
    p_mut: float = 0.3,
) -> MutResult:
    """
    Add N(0, sigma) noise to each gene with probability p_mut.
    """
    n = len(x)
    y = np.array(x, dtype=float)
    do_mut = rng.random(n) < p_mut
    noise = rng.normal(0.0, sigma, size=n)
    y2 = y.copy()
    y2[do_mut] = y2[do_mut] + noise[do_mut]
    changed = do_mut.tolist()
    return MutResult(mutated=y2.tolist(), changed=changed, info=f"gaussian σ={sigma}, p={p_mut}")


def shuffle_mutation(x: Sequence[float], start: int = 0, end: int=-1) -> MutResult:
    """
    Shuffle a slice x[start:end] in-place (end exclusive).
    """
    n = len(x)
    if not (0 <= start < end <= n):
        raise ValueError("Invalid shuffle slice.")
    y = list(x)
    before = y[start:end]
    after = before.copy()
    rng.shuffle(after)
    y[start:end] = after
    changed = [False] * n
    # Mark changed positions (even if some values coincidentally stay)
    for k in range(start, end):
        changed[k] = True
    return MutResult(mutated=y, changed=changed, info=f"shuffle [{start}:{end}] (len={end-start})")


# ---------------------------
# Demo
# ---------------------------

if __name__ == "__main__":
    parent_a = list(range(1, 11))          # 1..10
    parent_b = list(range(10, 0, -1))      # 10..1

    log("[bold]Parents[/bold]")
    log("A =", rich_vector(parent_a, styles=["green"] * len(parent_a)))
    log("B =", rich_vector(parent_b, styles=["blue"] * len(parent_b)))
    log("")

    # Crossover palette: 0->green (from A), 1->blue (from B)
    palette = ["green", "blue"]

    log("[bold]Crossover[/bold]")

    # One-point
    cx1 = one_point_crossover(parent_a, parent_b, point=5)
    log(f"[bold]One-Point[/bold] ({cx1.info})")
    log("child =", rich_vector(cx1.child, styles=segment_styles_from_sources(len(cx1.child), cx1.sources, palette)))
    log("")

    # N-point
    cxn = n_point_crossover(parent_a, parent_b, points=[3, 7])
    log(f"[bold]N-Point[/bold] ({cxn.info})")
    log("child =", rich_vector(cxn.child, styles=segment_styles_from_sources(len(cxn.child), cxn.sources, palette)))
    log("")

    # Uniform
    cxu = uniform_crossover(parent_a, parent_b, p_from_a=0.5)
    log(f"[bold]Uniform[/bold] ({cxu.info})")
    log("child =", rich_vector(cxu.child, styles=segment_styles_from_sources(len(cxu.child), cxu.sources, palette)))
    log("")

    log("[bold]Mutation[/bold]")

    base = cx1.child  # mutate the one-point child for a consistent example
    log("Base child =", rich_vector(base, styles=segment_styles_from_sources(len(base), cx1.sources, palette)))
    log("")

    # Swap
    m1 = swap_mutation(base, i=1, j=8)
    swap_styles = ["red" if ch else "dim" for ch in m1.changed]
    log(f"[bold]Swap[/bold] ({m1.info})  [dim](changed genes in red)[/dim]")
    log("mutated   =", rich_vector(m1.mutated, styles=swap_styles))
    log("")

    # Gaussian
    m2 = gaussian_mutation(base, sigma=0.75, p_mut=0.4)
    gauss_styles = ["red" if ch else "dim" for ch in m2.changed]
    log(f"[bold]Gaussian[/bold] ({m2.info})  [dim](mutated genes in red)[/dim]")
    log("mutated    =", rich_vector(m2.mutated, styles=gauss_styles))
    log("")

    # Shuffle
    m3 = shuffle_mutation(base, start=2, end=7)
    shuf_styles = ["red" if ch else "dim" for ch in m3.changed]
    log(f"[bold]Shuffle[/bold] ({m3.info})  [dim](shuffled slice in red)[/dim]")
    log("mutated    =", rich_vector(m3.mutated, styles=shuf_styles))