"""Custom plot registry for the ARIEL dashboard.

This module is intentionally standalone so that custom plot files can
``from ariel.visualisation.dashboard.plot_registry import custom_registry``
without pulling in the full dashboard (and avoiding circular imports).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class CustomPlotEntry:
    """Metadata for a single registered custom plot."""

    name: str
    columns: list[str]
    callback: Callable
    description: str = ""
    mode: str = "per_generation"  # "per_generation" | "full_figure"
    y_label: str = ""


class CustomPlotRegistry:
    """Registry for user-defined dashboard plots.

    Example — per-generation mode (returns scalar metrics, auto-plotted)::

        from ariel.visualisation.dashboard.plot_registry import custom_registry

        @custom_registry.register(
            name="Selection Pressure",
            columns=["fitness_", "n_offspring"],
            y_label="Pressure",
        )
        def selection_pressure(generation, data):
            return {"pressure": float(np.mean(data["n_offspring"])
                                      / (np.mean(data["fitness_"]) + 1e-9))}

    Example — full-figure mode (returns a go.Figure)::

        @custom_registry.register(
            name="My Plot",
            columns=["fitness_"],
            mode="full_figure",
        )
        def my_plot(generations, per_gen_data):
            import plotly.graph_objects as go
            fig = go.Figure()
            means = [float(np.mean(d["fitness_"])) for d in per_gen_data]
            fig.add_trace(go.Scatter(x=list(generations), y=means))
            return fig
    """

    def __init__(self) -> None:
        self._entries: dict[str, CustomPlotEntry] = {}

    # -- public API -----------------------------------------------------------

    def register(
        self,
        *,
        name: str,
        columns: list[str],
        description: str = "",
        mode: str = "per_generation",
        y_label: str = "",
    ) -> Callable:
        """Decorator that registers a custom plot function.

        Parameters
        ----------
        name:
            Display name shown in the sidebar button and page title.
        columns:
            Column names from the ``individual`` table that the callback
            needs (e.g. ``["fitness_", "genotype_"]``).
            ``time_of_birth`` and ``time_of_death`` are always fetched
            automatically — no need to list them.
        description:
            Optional subtitle shown on the page.
        mode:
            ``"per_generation"`` — callback signature::

                def fn(generation: int, data: dict[str, np.ndarray]) -> dict[str, float]

            ``"full_figure"`` — callback signature::

                def fn(generations: np.ndarray,
                       per_gen_data: list[dict[str, np.ndarray]]) -> go.Figure

        y_label:
            Y-axis label (only used in ``per_generation`` mode).
        """

        def decorator(fn: Callable) -> Callable:
            self._entries[name] = CustomPlotEntry(
                name=name,
                columns=columns,
                callback=fn,
                description=description,
                mode=mode,
                y_label=y_label or name,
            )
            return fn

        return decorator

    @property
    def entries(self) -> dict[str, CustomPlotEntry]:
        return self._entries

    # -- data fetching --------------------------------------------------------

    @staticmethod
    def query_columns(
        path: str, columns: list[str], sentinel: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Fetch requested columns plus birth/death from the DB.

        Returns (births, deaths, col_arrays) where col_arrays maps each
        requested column name to a NumPy array.
        """
        import sqlite3

        all_cols = ["time_of_birth", "time_of_death"] + [
            c for c in columns if c not in ("time_of_birth", "time_of_death")
        ]
        col_sql = ", ".join(all_cols)
        with sqlite3.connect(path) as conn:
            rows = conn.execute(
                f"SELECT {col_sql} FROM individual"  # noqa: S608
            ).fetchall()

        births = np.array(
            [r[0] for r in rows], dtype=np.int32,
        )
        deaths = np.array(
            [r[1] if r[1] is not None else sentinel for r in rows],
            dtype=np.int32,
        )

        col_arrays: dict[str, np.ndarray] = {}
        for idx, col_name in enumerate(all_cols):
            if col_name in ("time_of_birth", "time_of_death"):
                continue
            raw = [r[idx] for r in rows]
            # Attempt numeric conversion; fall back to object array.
            try:
                col_arrays[col_name] = np.array(
                    [float(v) if v is not None else np.nan for v in raw],
                    dtype=float,
                )
            except (ValueError, TypeError):
                col_arrays[col_name] = np.array(raw, dtype=object)

        return births, deaths, col_arrays

    # -- compute per-registered-plot ------------------------------------------

    def compute_custom(
        self, entry: CustomPlotEntry, path: str, min_gen: int, max_gen: int,
    ) -> dict:
        """Run a custom plot's callback against the DB data.

        For ``per_generation`` mode returns::
            {"generations": np.ndarray, "<metric>": np.ndarray, ...}

        For ``full_figure`` mode returns::
            {"figure": go.Figure}
        """
        sentinel = max_gen + 1
        births, deaths, col_arrays = self.query_columns(
            path, entry.columns, sentinel,
        )

        generations = np.arange(min_gen, max_gen + 1)

        if entry.mode == "full_figure":
            per_gen_data: list[dict[str, np.ndarray]] = []
            for gen in generations:
                alive = (births <= gen) & (deaths > gen)
                gen_data = {k: v[alive] for k, v in col_arrays.items()}
                per_gen_data.append(gen_data)
            fig = entry.callback(generations, per_gen_data)
            return {"figure": fig}

        # per_generation mode: collect scalar dicts across generations
        metric_lists: dict[str, list[float]] = {}
        for gen in generations:
            alive = (births <= gen) & (deaths > gen)
            if not alive.any():
                if metric_lists:
                    for k in metric_lists:
                        metric_lists[k].append(float("nan"))
                continue
            gen_data = {k: v[alive] for k, v in col_arrays.items()}
            result = entry.callback(int(gen), gen_data)
            for k, v in result.items():
                metric_lists.setdefault(k, []).append(float(v))

        out: dict[str, np.ndarray] = {"generations": generations}
        for k, vals in metric_lists.items():
            out[k] = np.asarray(vals)
        return out


# Module-level singleton — the single entry point for all registrations.
custom_registry = CustomPlotRegistry()
