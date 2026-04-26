---
type: api_reference
tags: [mujoco, python, class, visualization, plotting]
source: https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjvisualize.h
date_ingested: 2026-04-13
---
# MjvFigure

2D figure / plot overlay rendered into a viewport by `mjr_figure`. Supports up to 100 line series, bar plots, legends, and custom axis formatting.

## Signature

```python
fig = mujoco.MjvFigure()
mujoco.mjv_defaultFigure(fig)   # initialize to defaults
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `flg_legend` | int | Show legend (1 = yes) |
| `flg_ticklabel` | `(2,)` int | Show tick labels: [x-axis, y-axis] |
| `flg_extend` | int | Auto-extend ranges to fit data |
| `flg_barplot` | int | Bar plot mode (1) vs line plot (0) |
| `flg_selection` | int | Draw vertical selection line |
| `flg_symmetric` | int | Symmetric y-axis around zero |
| `linewidth` | float | Line width in pixels |
| `gridwidth` | float | Grid line width in pixels |
| `gridsize` | `(2,)` int | Number of grid cells: [x, y] |
| `gridrgb` | `(3,)` float | Grid line color [R, G, B] |
| `figurergba` | `(4,)` float | Figure background [R, G, B, A] |
| `panergba` | `(4,)` float | Pane (plot area) background [R, G, B, A] |
| `legendrgba` | `(4,)` float | Legend background [R, G, B, A] |
| `textrgb` | `(3,)` float | Text color [R, G, B] |
| `linergb` | `(100, 3)` float | Line colors per series |
| `range` | `(2, 2)` float | Axis ranges: `[[xmin, xmax], [ymin, ymax]]` |
| `xformat` | str (char[20]) | x-axis tick format string (e.g. `"%.2f"`) |
| `yformat` | str (char[20]) | y-axis tick format string |
| `minwidth` | str (char[20]) | Minimum width string for label sizing |
| `title` | str (char[1000]) | Figure title |
| `xlabel` | str (char[100]) | x-axis label |
| `linename` | `(100,)` str | Legend names per series (char[100] each) |
| `legendoffset` | int | Legend vertical offset |
| `subplot` | int | Subplot index (for multi-panel layouts) |
| `highlight` | `(2,)` int | Highlight rectangle |
| `highlightid` | int | Highlight id |
| `selection` | float | x-value of vertical selection line |
| `linepnt` | `(100,)` int | Number of data points per line series |
| `linedata` | `(100, 2002)` float | Line data: interleaved [x0, y0, x1, y1, …] per series (max 1001 pts) |
| `xaxispixel` | `(2,)` int | x-axis pixel extent [left, right] |
| `yaxispixel` | `(2,)` int | y-axis pixel extent [bottom, top] |
| `xaxisdata` | `(2,)` float | x-axis data range [min, max] |
| `yaxisdata` | `(2,)` float | y-axis data range [min, max] |

## Usage

```python
import mujoco
import numpy as np

fig = mujoco.MjvFigure()
mujoco.mjv_defaultFigure(fig)

# Configure
fig.title    = "Reward"
fig.xlabel   = "Step"
fig.flg_legend = 1
fig.linename[0] = "episode"
fig.linergb[0]  = np.array([0.0, 1.0, 0.2])   # green
fig.range[0]    = [0.0, 1000.0]                # x range
fig.range[1]    = [-1.0, 1.0]                  # y range

# Add a data point to line 0
n = fig.linepnt[0]          # current point count
if n < 1001:
    fig.linedata[0, n*2]     = step_number      # x
    fig.linedata[0, n*2 + 1] = reward_value     # y
    fig.linepnt[0] += 1

# Render into a sub-viewport (top-right corner example)
fig_viewport = mujoco.MjrRect(width - 300, height - 200, 300, 200)
mujoco.mjr_figure(fig_viewport, fig, ctx)
```

## Notes

- `linedata` holds at most 1001 points per series (indices 0–1000, interleaved x/y → 2002 floats).
- `mjr_figure` must be called **after** `mjr_render` so the figure overlays the 3D scene.
- When `flg_extend = 1`, `range` is overridden automatically to fit data.
- To clear a series, set `fig.linepnt[i] = 0`.

See [[mujoco_visualization_functions]], [[MjrContext]], [[MjrRect]].
