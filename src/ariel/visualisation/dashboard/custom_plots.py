"""Custom plots for the ARIEL dashboard.

Place this file as ``custom_plots.py`` next to ``dash_fast_mod.py``
(or put individual files inside a ``custom_plots/`` sub-directory).

Import ``custom_registry`` from ``plot_registry`` (NOT from
``dash_fast_mod``) to avoid circular-import issues.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from ariel.visualisation.dashboard.plot_registry import custom_registry


# # Per-generation mode (simplest — returns scalar metrics, auto-plotted as lines)
# @custom_registry.register(
#     name="Selection Pressure",
#     columns=["fitness_", "n_offspring"],
#     y_label="Pressure",
# )
# def selection_pressure(generation, data):
#     return {"pressure": float(np.mean(data["n_offspring"]) / (np.mean(data["fitness_"]) + 1e-9))}



# Full-figure mode (full control — return your own go.Figure)
@custom_registry.register(
    name="My Custom Plot",
    columns=["fitness_"],
    mode="full_figure",
)
def my_plot(generations, per_gen_data):
    fig = go.Figure()
    means = [float(np.mean(d["fitness_"])) for d in per_gen_data]
    fig.add_trace(go.Scatter(x=list(generations), y=means, name="Mean"))
    return fig