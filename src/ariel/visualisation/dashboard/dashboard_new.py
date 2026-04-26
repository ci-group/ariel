"""ARIEL Evolution Dashboard.

Interactive Panel dashboard for visualising evolutionary experiment data
stored in an SQLite database (ARIEL format).

Usage:
    panel serve dashboard_new.py --show
"""

import io
import pathlib
import sqlite3
import tempfile
import warnings

import json5
import numpy as np
import pandas as pd
import panel as pn
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Colour themes  (each entry: list of hex colours assigned to traces in order)
# ─────────────────────────────────────────────────────────────────────────────
_COLOUR_THEMES: dict[str, list[str]] = {
    "ARIEL (default)": [
        "#1f77b4",
        "#ff7f0e",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#8c6d31",
    ],
    "Viridis": [
        "#440154",
        "#31688e",
        "#35b779",
        "#fde725",
        "#21918c",
        "#5ec962",
    ],
    "Warm": ["#e63946", "#f4a261", "#e9c46a", "#2a9d8f", "#264653", "#a8dadc"],
    "Cool": ["#023e8a", "#0077b6", "#0096c7", "#00b4d8", "#48cae4", "#90e0ef"],
    "Pastel": [
        "#6a9fb5",
        "#c47c5a",
        "#8aba6a",
        "#9b6bb5",
        "#c4a05a",
        "#5ab5a0",
    ],
    "Monochrome": [
        "#222222",
        "#555555",
        "#888888",
        "#aaaaaa",
        "#cccccc",
        "#e8e8e8",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Global plot-style widgets  (referenced by _apply_theme at call-time)
# ─────────────────────────────────────────────────────────────────────────────
sld_font_size = pn.widgets.IntSlider(
    name="Font size",
    value=12,
    start=8,
    end=24,
    step=1,
)
sld_fig_width = pn.widgets.Select(
    name="PDF export aspect ratio",
    options={
        "Horizontal (2:1)": (1200, 600),
        "Vertical (1:2)": (600, 1200),
        "Square (1:1)": (900, 900),
    },
    value=(1200, 600),
)
sel_colour_theme = pn.widgets.Select(
    name="Colour theme",
    options=list(_COLOUR_THEMES.keys()),
    value="ARIEL (default)",
)


# ─────────────────────────────────────────────────────────────────────────────
# Plotly theme helpers
# ─────────────────────────────────────────────────────────────────────────────
def _apply_theme(fig: go.Figure, font_size: int) -> go.Figure:
    """Apply the shared Plotly layout theme with user-chosen font size."""
    title_size = font_size + 4
    tick_size = max(font_size - 1, 8)
    legend_size = max(font_size - 1, 8)

    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafd",
        font={
            "family": "Segoe UI, system-ui, sans-serif",
            "color": "#374151",
            "size": font_size,
        },
        title_font={
            "size": title_size,
            "color": "#1f2e45",
            "family": "Segoe UI, system-ui, sans-serif",
        },
        xaxis={
            "gridcolor": "#e2e8f4",
            "gridwidth": 1,
            "linecolor": "#d0d9e8",
            "tickfont": {"color": "#6b7280", "size": tick_size},
            "title_font": {"color": "#374151", "size": font_size},
            "showgrid": True,
        },
        yaxis={
            "gridcolor": "#e2e8f4",
            "gridwidth": 1,
            "linecolor": "#d0d9e8",
            "tickfont": {"color": "#6b7280", "size": tick_size},
            "title_font": {"color": "#374151", "size": font_size},
            "showgrid": True,
        },
        legend={
            "bgcolor": "rgba(255,255,255,0.95)",
            "bordercolor": "#d0d9e8",
            "borderwidth": 1,
            "font": {"size": legend_size, "color": "#374151"},
        },
        hovermode="x unified",
        margin={"l": 60, "r": 30, "t": 60, "b": 50},
    )
    return fig


def _wrap_plotly(fig: go.Figure, font_size: int) -> pn.pane.Plotly:
    """Apply theme and wrap a Plotly figure in a Panel pane."""
    return pn.pane.Plotly(
        _apply_theme(fig, font_size),
        sizing_mode="stretch_width",
        height=480,
    )


def _pdf_download_btn(fig_getter, filename: str) -> pn.widgets.FileDownload:
    """Return a FileDownload button that exports the current figure as a PDF.

    fig_getter: zero-argument callable returning a go.Figure (or None).
    Requires kaleido: pip install kaleido
    """

    def _callback() -> io.BytesIO:
        fig = fig_getter()
        if fig is None:
            return io.BytesIO(b"")
        _apply_theme(fig, sld_font_size.value)
        w, h = sld_fig_width.value
        buf = io.BytesIO()
        fig.write_image(buf, format="pdf", width=w, height=h)
        buf.seek(0)
        return buf

    return pn.widgets.FileDownload(
        callback=_callback,
        filename=filename,
        button_type="default",
        icon="file-download",
        label=" Save as PDF",
        styles={"margin-top": "6px"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSS theme
# ─────────────────────────────────────────────────────────────────────────────
_CSS = """
/* ── Global ── */
body, html {
    font-family: 'Segoe UI', system-ui, sans-serif !important;
    background: #eef1f8 !important;
}

/* ── Top navbar ── */
.navbar {
    background: linear-gradient(135deg, #1f2e45 0%, #2d4163 100%) !important;
    box-shadow: 0 2px 12px rgba(31,46,69,0.25) !important;
}
.navbar-brand {
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    color: #e8edf8 !important;
}

/* ── Sidebar ── */
#sidebar {
    background: #ffffff !important;
    border-right: 1px solid #dce6f4 !important;
    box-shadow: 3px 0 16px rgba(31,46,69,0.07) !important;
}

/* Sidebar section headings */
#sidebar h2 {
    color: #1f2e45 !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    padding-bottom: 6px !important;
    border-bottom: 2px solid #4f8ef7 !important;
    display: inline-block !important;
}
#sidebar h3 {
    color: #64748b !important;
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    margin: 4px 0 6px 2px !important;
}

/* Sidebar nav buttons */
#sidebar .bk-btn {
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-align: left !important;
    padding: 9px 14px !important;
    margin-bottom: 3px !important;
    border: 1px solid transparent !important;
    transition: background 0.15s, color 0.15s, transform 0.12s, box-shadow 0.15s !important;
}
#sidebar .bk-btn-warning {
    background: #f0f4ff !important;
    color: #1f2e45 !important;
    border-color: #dce6f7 !important;
}
#sidebar .bk-btn-warning:hover {
    background: #1f2e45 !important;
    color: #ffffff !important;
    border-color: #1f2e45 !important;
    transform: translateX(4px) !important;
    box-shadow: 0 3px 10px rgba(31,46,69,0.20) !important;
}

/* Sidebar divider */
.bk-Divider {
    border-top: 1px solid #e8edf5 !important;
    margin: 10px 0 !important;
}

/* Sidebar checkboxes */
#sidebar .bk-input-group label {
    font-size: 0.82rem !important;
    color: #374151 !important;
    font-weight: 500 !important;
}
#sidebar input[type="checkbox"] {
    accent-color: #4f8ef7 !important;
}

/* ── Main content area ── */
#main {
    background: #eef1f8 !important;
    padding: 28px 32px !important;
}

/* Card wrapper around each page */
#main > .bk-panel-models-layout-Column > .bk-panel-models-layout-Column {
    background: #ffffff !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 20px rgba(31,46,69,0.09) !important;
    padding: 28px 32px !important;
    border: 1px solid #e4ecf7 !important;
}

/* Page titles */
#main h2 {
    color: #1f2e45 !important;
    font-weight: 700 !important;
    font-size: 1.3rem !important;
    margin-top: 0 !important;
    margin-bottom: 4px !important;
}

/* Subtitle / description text */
#main p {
    color: #4a5568 !important;
    font-size: 0.88rem !important;
    line-height: 1.6 !important;
}

/* ── FileInput browse button ── */
.bk-btn-default {
    border-radius: 8px !important;
    border: 1.5px dashed #a0b4d0 !important;
    background: #f5f8ff !important;
    color: #1f2e45 !important;
    font-weight: 600 !important;
    padding: 10px 18px !important;
    transition: border-color 0.15s, background 0.15s !important;
}
.bk-btn-default:hover {
    border-color: #4f8ef7 !important;
    background: #ebf2ff !important;
}

/* ── Primary "Load selected file" button ── */
.bk-btn-primary {
    background: linear-gradient(135deg, #1f2e45 0%, #2d4163 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    color: #fff !important;
    box-shadow: 0 2px 8px rgba(31,46,69,0.20) !important;
    transition: box-shadow 0.15s, transform 0.12s !important;
}
.bk-btn-primary:hover {
    box-shadow: 0 5px 16px rgba(31,46,69,0.30) !important;
    transform: translateY(-1px) !important;
}
"""

pn.extension("plotly", sizing_mode="stretch_width", raw_css=[_CSS])

# ─────────────────────────────────────────────────────────────────────────────
# Global mutable state  — keyed by db label
# Each entry: {"df": DataFrame, "stats": dict}
# ─────────────────────────────────────────────────────────────────────────────
_databases: dict[str, dict] = {}

# DB-selector widget (populated dynamically as DBs are loaded)
mc_db_select = pn.widgets.MultiChoice(
    name="Active databases",
    options=[],
    value=[],
    placeholder="Load a database first…",
)
chk_show_individual = pn.widgets.Checkbox(
    name="Show individual databases",
    value=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core computation  (runs once per load)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_stats(df: pd.DataFrame) -> dict:
    """Pre-compute all per-generation statistics from the database."""
    if df["time_of_death"].isna().any():
        df = df.copy()
        df["time_of_death"] = df["time_of_death"].fillna(
            df["time_of_death"].max() + 1,
        )

    min_gen = int(df["time_of_birth"].min())
    max_gen = int(df["time_of_death"].max())

    id_to_geno = dict(zip(df["id"], df["genotype_"], strict=False))
    id_to_fitness = df.set_index("id")["fitness_"]
    id_to_birth = df.set_index("id")["time_of_birth"]

    generations = list(range(min_gen, max_gen + 1))

    fit_mean, fit_std, fit_max, fit_min = [], [], [], []
    diversity = []
    age_mean, age_max, age_median = [], [], []

    for gen in generations:
        alive_mask = (df["time_of_birth"] <= gen) & (df["time_of_death"] > gen)
        alive_ids = df.loc[alive_mask, "id"].tolist()

        if not alive_ids:
            for lst in (
                fit_mean,
                fit_std,
                fit_max,
                fit_min,
                diversity,
                age_mean,
                age_max,
                age_median,
            ):
                lst.append(float("nan"))
            continue

        # ── Fitness ──────────────────────────────────────────────────────────
        fits = id_to_fitness.reindex(alive_ids).dropna().astype(float).values  # noqa: PD011
        if fits.size:
            fit_mean.append(float(np.mean(fits)))
            fit_std.append(float(np.std(fits, ddof=0)))
            fit_max.append(float(np.max(fits)))
            fit_min.append(float(np.min(fits)))
        else:
            for lst in (fit_mean, fit_std, fit_max, fit_min):
                lst.append(float("nan"))

        # ── Genotype matrix ──────────────────────────────────────────────────
        G = np.array([
            np.asarray(id_to_geno[i], dtype=float) for i in alive_ids
        ])
        n = len(alive_ids)

        if n > 1:
            diff = G[:, None, :] - G[None, :, :]
            diag = np.sqrt(np.sum(diff**2, axis=-1))
            triu = diag[np.triu_indices(n, k=1)]
            diversity.append(float(triu.mean()))
        else:
            diversity.append(0.0)

        # ── Age ───────────────────────────────────────────────────────────────
        ages = np.array([gen - id_to_birth[i] for i in alive_ids], dtype=float)
        age_mean.append(float(np.mean(ages)))
        age_max.append(float(np.max(ages)))
        age_median.append(float(np.median(ages)))

    return {
        "generations": np.asarray(generations),
        "fit_mean": np.asarray(fit_mean),
        "fit_std": np.asarray(fit_std),
        "fit_max": np.asarray(fit_max),
        "fit_min": np.asarray(fit_min),
        "diversity": np.asarray(diversity),
        "age_mean": np.asarray(age_mean),
        "age_max": np.asarray(age_max),
        "age_median": np.asarray(age_median),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Novelty computation  (separate — depends on user-chosen window size)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_novelty(df: pd.DataFrame, window: int) -> dict:
    """Compute per-generation novelty using the N previous gens as archive.

    For each generation `gen`:
      - Current population : individuals alive at `gen`
      - Archive            : individuals alive at any of gen-window … gen-1
      - N(x, archive)     : mean Euclidean distance from individual x to every
                            individual in the archive
      - nov_mean / nov_max: mean and max of N(x, archive) over the current pop.

    When the archive is empty (first `window` generations) we emit NaN.

    Returns
    -------
    dict
    """
    if df["time_of_death"].isna().any():
        df = df.copy()
        df["time_of_death"] = df["time_of_death"].fillna(
            df["time_of_death"].max() + 1,
        )

    id_to_geno = dict(zip(df["id"], df["genotype_"], strict=False))
    generations = list(
        range(
            int(df["time_of_birth"].min()), int(df["time_of_death"].max()) + 1,
        ),
    )

    nov_mean, nov_max = [], []

    for gen in generations:
        # Current population
        cur_mask = (df["time_of_birth"] <= gen) & (df["time_of_death"] > gen)
        cur_ids = df.loc[cur_mask, "id"].tolist()

        # Archive: individuals alive in any of the previous `window` generations
        arch_mask = (df["time_of_birth"] <= gen - 1) & (
            df["time_of_death"] > gen - window
        )
        arch_ids = df.loc[arch_mask, "id"].tolist()

        if not cur_ids or not arch_ids:
            nov_mean.append(float("nan"))
            nov_max.append(float("nan"))
            continue

        g_cur = np.array([
            np.asarray(id_to_geno[i], dtype=float) for i in cur_ids
        ])
        g_arch = np.array([
            np.asarray(id_to_geno[i], dtype=float) for i in arch_ids
        ])

        # Distance matrix: (n_cur, n_arch)
        diff = g_cur[:, None, :] - g_arch[None, :, :]  # (n_cur, n_arch, dims)
        diag = np.sqrt(np.sum(diff**2, axis=-1))  # (n_cur, n_arch)

        nov_i = diag.mean(axis=1)  # mean dist to archive per individual
        nov_mean.append(float(np.mean(nov_i)))
        nov_max.append(float(np.max(nov_i)))

    return {
        "generations": np.asarray(generations),
        "nov_mean": np.asarray(nov_mean),
        "nov_max": np.asarray(nov_max),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers  (average across selected DBs on a union generation axis)
# ─────────────────────────────────────────────────────────────────────────────
def _aggregate_stats(selected_dbs: list) -> "dict | None":
    """Return nanmean of each stat key across all selected databases.

    DBs may span different generation ranges; values outside a DB's range
    are treated as NaN so they don't drag the mean down.

    Returns
    -------
    dict | None
    """
    if not selected_dbs:
        return None

    stat_keys = [
        "fit_mean",
        "fit_std",
        "fit_max",
        "fit_min",
        "diversity",
        "age_mean",
        "age_max",
        "age_median",
    ]

    # Collect one Series per DB per key, all indexed by generation
    series: dict[str, list] = {k: [] for k in stat_keys}
    all_gens: set = set()

    for label in selected_dbs:
        entry = _databases.get(label)
        if entry is None:
            continue
        s = entry["stats"]
        gens = s["generations"].astype(int)
        all_gens.update(gens.tolist())
        for k in stat_keys:
            series[k].append(pd.Series(s[k], index=gens, dtype=float))

    if not all_gens:
        return None

    union_gens = np.array(sorted(all_gens))

    result: dict = {"generations": union_gens}
    for k in stat_keys:
        if not series[k]:
            result[k] = np.full(len(union_gens), np.nan)
        else:
            df_aligned = pd.concat(series[k], axis=1).reindex(union_gens)
            result[k] = df_aligned.mean(axis=1, skipna=True).to_numpy()

    return result


def _aggregate_novelty(selected_dbs: list, window: int) -> "dict | None":
    """Return nanmean novelty across all selected databases.

    Returns
    -------
    dict
    """
    if not selected_dbs:
        return None

    nov_mean_series, nov_max_series = [], []
    all_gens: set = set()

    for label in selected_dbs:
        entry = _databases.get(label)
        if entry is None:
            continue
        nov = _compute_novelty(entry["df"], window)
        gens = nov["generations"].astype(int)
        all_gens.update(gens.tolist())
        nov_mean_series.append(
            pd.Series(nov["nov_mean"], index=gens, dtype=float),
        )
        nov_max_series.append(
            pd.Series(nov["nov_max"], index=gens, dtype=float),
        )

    if not all_gens:
        return None

    union_gens = np.array(sorted(all_gens))
    mean_agg = (
        pd.concat(nov_mean_series, axis=1)
        .reindex(union_gens)
        .mean(axis=1, skipna=True)
        .to_numpy()
    )
    max_agg = (
        pd.concat(nov_max_series, axis=1)
        .reindex(union_gens)
        .mean(axis=1, skipna=True)
        .to_numpy()
    )

    return {"generations": union_gens, "nov_mean": mean_agg, "nov_max": max_agg}


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def _no_data_msg() -> pn.pane.Markdown:
    return pn.pane.Markdown(
        "⚠️  **No data loaded.**  \n"
        "Please go to **Load Database** and open a `.db` file first.",
    )


def _no_selection_msg() -> pn.pane.Markdown:
    return pn.pane.Markdown(
        "⚠️  **No databases selected.**  \n"
        "Use the **Active databases** selector in the sidebar to choose at least one.",
    )


def _hex_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)'.

    Returns
    -------
    str
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _parse_yrange(ymin_str: str, ymax_str: str) -> tuple[float, float] | None:
    """Parse y-range textbox strings. Returns (min, max) or None for autorange.

    Returns
    -------
    tuple[float,float]
    """
    try:
        ymin = float(ymin_str) if ymin_str.strip() else None
        ymax = float(ymax_str) if ymax_str.strip() else None
        if ymin is not None and ymax is not None:
            return (ymin, ymax)
    except ValueError:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Load-Database page
# ─────────────────────────────────────────────────────────────────────────────
_file_sel = pn.widgets.FileInput(
    accept=".db",
    multiple=False,
    name="Click to open file explorer and select a .db file",
)
_status_md = pn.pane.Markdown("")
_load_btn = pn.widgets.Button(
    name="Load selected file",
    button_type="primary",
    styles={"margin-top": "8px"},
)


def _unique_label(base: str) -> str:
    """Return `base`, or `base (2)`, `base (3)` … if already taken.

    Returns
    -------
    str
    """
    if base not in _databases:
        return base
    i = 2
    while f"{base} ({i})" in _databases:
        i += 1
    return f"{base} ({i})"


def _on_load(event : None) -> None:
    global _databases
    if _file_sel.value is None:
        _status_md.object = (
            "⚠️ No file selected. Click the button above to pick a .db file."
        )
        return
    try:
        _status_md.object = "Loading…"
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            tmp.write(_file_sel.value)
            tmp_path = tmp.name
        conn = sqlite3.connect(tmp_path)
        try:
            df = pd.read_sql("SELECT * FROM individual", conn)
        finally:
            conn.close()
        pathlib.Path(tmp_path).unlink()
        df["genotype_"] = df["genotype_"].apply(json5.loads)

        label = _unique_label(_file_sel.filename or "database")
        stats = _compute_stats(df)
        _databases[label] = {"df": df, "stats": stats}

        # Update the selector — keep existing selections, add new entry
        mc_db_select.options = list(_databases.keys())
        mc_db_select.value = list(_databases.keys())  # auto-select all

        n_gen = int((stats["generations"][-1] - stats["generations"][0]) + 1)
        loaded_list = "\n".join(f"- **{k}**" for k in _databases)
        _status_md.object = (
            f"✅ Loaded **{label}** — {len(df):,} individuals, {n_gen} generations.\n\n"
            f"**All loaded databases:**\n{loaded_list}"
        )
    except Exception as exc:
        _status_md.object = f"❌ Error: {exc}"


_load_btn.on_click(_on_load)


def _make_load_page() -> pn.Column:
    return pn.Column(
        pn.pane.Markdown("## Load Database"),
        pn.pane.Markdown(
            "Select a `.db` file below and press **Load selected file**.  \n"
            "You can load multiple databases — they will all appear in the sidebar selector.",
        ),
        _file_sel,
        _load_btn,
        _status_md,
        align="center",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar per-plot toggle widgets
# ─────────────────────────────────────────────────────────────────────────────

# -- Fitness --
chk_fit_mean = pn.widgets.Checkbox(name="Mean", value=True)
chk_fit_std = pn.widgets.Checkbox(name="Std shading", value=True)
chk_fit_max = pn.widgets.Checkbox(name="Max", value=True)
chk_fit_min = pn.widgets.Checkbox(name="Min", value=False)
txt_fit_yaxis = pn.widgets.TextInput(
    name="Y-axis label", value="Fitness  f(x)", width=200,
)
txt_fit_ymin = pn.widgets.TextInput(name="Y min", placeholder="auto", width=95)
txt_fit_ymax = pn.widgets.TextInput(name="Y max", placeholder="auto", width=95)
chk_fit_hide_title = pn.widgets.Checkbox(name="Hide title", value=False)

# -- Novelty --
chk_nov_mean = pn.widgets.Checkbox(name="Mean", value=True)
chk_nov_max = pn.widgets.Checkbox(name="Max", value=True)
sld_nov_window = pn.widgets.IntSlider(
    name="Archive window (N prev. generations)",
    value=1,
    start=1,
    end=20,
    step=1,
)
txt_nov_yaxis = pn.widgets.TextInput(
    name="Y-axis label", value="Novelty  N(x, archive)", width=200,
)
txt_nov_ymin = pn.widgets.TextInput(name="Y min", placeholder="auto", width=95)
txt_nov_ymax = pn.widgets.TextInput(name="Y max", placeholder="auto", width=95)
chk_nov_hide_title = pn.widgets.Checkbox(name="Hide title", value=False)

# -- Diversity --
txt_div_yaxis = pn.widgets.TextInput(
    name="Y-axis label", value="Diversity  D(P)", width=200,
)
txt_div_ymin = pn.widgets.TextInput(name="Y min", placeholder="auto", width=95)
txt_div_ymax = pn.widgets.TextInput(name="Y max", placeholder="auto", width=95)
chk_div_hide_title = pn.widgets.Checkbox(name="Hide title", value=False)

# -- Age --
chk_age_mean = pn.widgets.Checkbox(name="Mean", value=True)
chk_age_max = pn.widgets.Checkbox(name="Max", value=True)
chk_age_median = pn.widgets.Checkbox(name="Median", value=True)
txt_age_yaxis = pn.widgets.TextInput(
    name="Y-axis label", value="Age  L(x)  [generations]", width=200,
)
txt_age_ymin = pn.widgets.TextInput(name="Y min", placeholder="auto", width=95)
txt_age_ymax = pn.widgets.TextInput(name="Y max", placeholder="auto", width=95)
chk_age_hide_title = pn.widgets.Checkbox(name="Hide title", value=False)


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 - Fitness over time  (fully reactive)
# ─────────────────────────────────────────────────────────────────────────────
def _fitness_fig(
    show_mean,  # noqa: ANN001
    show_std,
    show_max,
    show_min,
    theme,
    yaxis_label,
    ymin,
    ymax,
    hide_title,
    selected_dbs,
    show_individual,
) -> "go.Figure | None":
    if not selected_dbs:
        return None

    colors = _COLOUR_THEMES[theme]
    fig = go.Figure()
    yrange = _parse_yrange(ymin, ymax)

    if show_individual:
        for i, label in enumerate(selected_dbs):
            entry = _databases.get(label)
            if entry is None:
                continue
            s = entry["stats"]
            x = s["generations"]
            col = colors[i % len(colors)]
            if show_mean:
                if show_std:
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=s["fit_mean"] - s["fit_std"],
                            mode="lines",
                            line={"width": 0, "color": "rgba(255,255,255,0)"},
                            hoverinfo="skip",
                            showlegend=False,
                            legendgroup=label,
                            name=f"_lower_{label}",
                        ),
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=s["fit_mean"] + s["fit_std"],
                            mode="lines",
                            line={"width": 0, "color": "rgba(255,255,255,0)"},
                            fill="tonexty",
                            fillcolor=_hex_rgba(col, 0.15),
                            hoverinfo="skip",
                            showlegend=True,
                            legendgroup=label,
                            name=f"{label} — ±Std",
                        ),
                    )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=s["fit_mean"],
                        mode="lines",
                        line={"color": col, "width": 2},
                        legendgroup=label,
                        name=f"{label} — Mean",
                    ),
                )
            if show_max:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=s["fit_max"],
                        mode="lines",
                        line={"color": col, "width": 1.5, "dash": "dash"},
                        legendgroup=label,
                        name=f"{label} — Max",
                    ),
                )
            if show_min:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=s["fit_min"],
                        mode="lines",
                        line={"color": col, "width": 1.5, "dash": "dot"},
                        legendgroup=label,
                        name=f"{label} — Min",
                    ),
                )
    else:
        s = _aggregate_stats(selected_dbs)
        if s is None:
            return None
        x = s["generations"]
        col = colors[0]
        if show_mean:
            if show_std:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=s["fit_mean"] - s["fit_std"],
                        mode="lines",
                        line={"width": 0, "color": "rgba(255,255,255,0)"},
                        hoverinfo="skip",
                        showlegend=False,
                        name="_lower",
                    ),
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=s["fit_mean"] + s["fit_std"],
                        mode="lines",
                        line={"width": 0, "color": "rgba(255,255,255,0)"},
                        fill="tonexty",
                        fillcolor=_hex_rgba(col, 0.15),
                        hoverinfo="skip",
                        showlegend=True,
                        name="Mean ± Std",
                    ),
                )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=s["fit_mean"],
                    mode="lines",
                    line={"color": col, "width": 2},
                    name="Mean fitness",
                ),
            )
        if show_max:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=s["fit_max"],
                    mode="lines",
                    line={"color": colors[1], "width": 1.5, "dash": "dash"},
                    name="Max fitness",
                ),
            )
        if show_min:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=s["fit_min"],
                    mode="lines",
                    line={"color": colors[2], "width": 1.5, "dash": "dot"},
                    name="Min fitness",
                ),
            )

    fig.update_layout(
        title="" if hide_title else "Fitness statistics per generation",
        xaxis_title="Generation",
        yaxis_title=yaxis_label,
        yaxis={"range": list(yrange) if yrange else None},
    )
    return fig


def _fitness_plot(
    show_mean,
    show_std,
    show_max,
    show_min,
    theme,
    font_size,
    yaxis_label,
    ymin,
    ymax,
    hide_title,
    selected_dbs,
    show_individual,
):
    if not _databases:
        return _no_data_msg()
    if not selected_dbs:
        return _no_selection_msg()
    fig = _fitness_fig(
        show_mean,
        show_std,
        show_max,
        show_min,
        theme,
        yaxis_label,
        ymin,
        ymax,
        hide_title,
        selected_dbs,
        show_individual,
    )
    return _wrap_plotly(fig, font_size) if fig is not None else _no_data_msg()


def _make_fitness_page() -> pn.Column:
    return pn.Column(
        pn.pane.Markdown("## Fitness over Generations"),
        pn.pane.Markdown(
            "Aggregated population fitness **f(x)** per generation.  "
            "Toggle the **Fitness** options in the sidebar.",
        ),
        pn.Row(txt_fit_ymin, txt_fit_ymax),
        chk_fit_hide_title,
        pn.bind(
            _fitness_plot,
            show_mean=chk_fit_mean,
            show_std=chk_fit_std,
            show_max=chk_fit_max,
            show_min=chk_fit_min,
            theme=sel_colour_theme,
            font_size=sld_font_size,
            yaxis_label=txt_fit_yaxis,
            ymin=txt_fit_ymin,
            ymax=txt_fit_ymax,
            hide_title=chk_fit_hide_title,
            selected_dbs=mc_db_select,
            show_individual=chk_show_individual,
        ),
        _pdf_download_btn(
            lambda: _fitness_fig(
                chk_fit_mean.value,
                chk_fit_std.value,
                chk_fit_max.value,
                chk_fit_min.value,
                sel_colour_theme.value,
                txt_fit_yaxis.value,
                txt_fit_ymin.value,
                txt_fit_ymax.value,
                chk_fit_hide_title.value,
                mc_db_select.value,
                chk_show_individual.value,
            ),
            "fitness.pdf",
        ),
        align="center",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 - Diversity over time  (single line, no toggles)
# ─────────────────────────────────────────────────────────────────────────────
def _diversity_fig(theme,
                   yaxis_label,
                   ymin,
                   ymax,
                   hide_title,
                   selected_dbs,
                   show_individual,
) -> go.Figure | None:
    if not selected_dbs:
        return None
    colors = _COLOUR_THEMES[theme]
    fig = go.Figure()
    yrange = _parse_yrange(ymin, ymax)

    if show_individual:
        for i, label in enumerate(selected_dbs):
            entry = _databases.get(label)
            if entry is None:
                continue
            s = entry["stats"]
            col = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=s["generations"],
                    y=s["diversity"],
                    mode="lines",
                    line={"color": col, "width": 2},
                    legendgroup=label,
                    name=f"{label} — D(P)",
                ),
            )
    else:
        s = _aggregate_stats(selected_dbs)
        if s is None:
            return None
        fig.add_trace(
            go.Scatter(
                x=s["generations"],
                y=s["diversity"],
                mode="lines",
                line={"color": colors[0], "width": 2},
                name="Mean D(P)",
            ),
        )

    fig.update_layout(
        title="" if hide_title else "Population diversity over generations",
        xaxis_title="Generation",
        yaxis_title=yaxis_label,
        yaxis={"range": list(yrange) if yrange else None},
    )
    return fig


def _diversity_plot(
    theme,
    font_size,
    yaxis_label,
    ymin,
    ymax,
    hide_title,
    selected_dbs,
    show_individual,
):
    if not _databases:
        return _no_data_msg()
    if not selected_dbs:
        return _no_selection_msg()
    fig = _diversity_fig(
        theme,
        yaxis_label,
        ymin,
        ymax,
        hide_title,
        selected_dbs,
        show_individual,
    )
    return _wrap_plotly(fig, font_size) if fig is not None else _no_data_msg()


def _make_diversity_page() -> pn.Column:
    return pn.Column(
        pn.pane.Markdown("## Diversity over Generations"),
        pn.pane.Markdown(
            "**D(P)** = average Euclidean distance between all pairs of genotypes "
            "in the population at each generation.",
        ),
        pn.Row(txt_div_ymin, txt_div_ymax),
        chk_div_hide_title,
        pn.bind(
            _diversity_plot,
            theme=sel_colour_theme,
            font_size=sld_font_size,
            yaxis_label=txt_div_yaxis,
            ymin=txt_div_ymin,
            ymax=txt_div_ymax,
            hide_title=chk_div_hide_title,
            selected_dbs=mc_db_select,
            show_individual=chk_show_individual,
        ),
        _pdf_download_btn(
            lambda: _diversity_fig(
                sel_colour_theme.value,
                txt_div_yaxis.value,
                txt_div_ymin.value,
                txt_div_ymax.value,
                chk_div_hide_title.value,
                mc_db_select.value,
                chk_show_individual.value,
            ),
            "diversity.pdf",
        ),
        align="center",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 - Novelty over time  (reactive — window slider + show toggles)
# ─────────────────────────────────────────────────────────────────────────────
def _novelty_fig(
    show_mean,
    show_max,
    window,
    theme,
    yaxis_label,
    ymin,
    ymax,
    hide_title,
    selected_dbs,
    show_individual,
) -> go.Figure | None:
    if not selected_dbs:
        return None
    colors = _COLOUR_THEMES[theme]
    fig = go.Figure()
    yrange = _parse_yrange(ymin, ymax)

    if show_individual:
        for i, label in enumerate(selected_dbs):
            entry = _databases.get(label)
            if entry is None:
                continue
            nov = _compute_novelty(entry["df"], window)
            col = colors[i % len(colors)]
            if show_mean:
                fig.add_trace(
                    go.Scatter(
                        x=nov["generations"],
                        y=nov["nov_mean"],
                        mode="lines",
                        line={"color": col, "width": 2},
                        legendgroup=label,
                        name=f"{label} — Mean",
                    ),
                )
            if show_max:
                fig.add_trace(
                    go.Scatter(
                        x=nov["generations"],
                        y=nov["nov_max"],
                        mode="lines",
                        line={"color": col, "width": 1.5, "dash": "dash"},
                        legendgroup=label,
                        name=f"{label} — Max",
                    ),
                )
    else:
        nov = _aggregate_novelty(selected_dbs, window)
        if nov is None:
            return None
        if show_mean:
            fig.add_trace(
                go.Scatter(
                    x=nov["generations"],
                    y=nov["nov_mean"],
                    mode="lines",
                    line={"color": colors[0], "width": 2},
                    name="Mean novelty",
                ),
            )
        if show_max:
            fig.add_trace(
                go.Scatter(
                    x=nov["generations"],
                    y=nov["nov_max"],
                    mode="lines",
                    line={"color": colors[1], "width": 1.5, "dash": "dash"},
                    name="Max novelty",
                ),
            )

    fig.update_layout(
        title=""
        if hide_title
        else f"Individual novelty over generations  (archive = {window} prev. generation{'s' if window > 1 else ''})",
        xaxis_title="Generation",
        yaxis_title=yaxis_label,
        yaxis={"range": list(yrange) if yrange else None},
    )
    return fig


def _novelty_plot(
    show_mean,
    show_max,
    window,
    theme,
    font_size,
    yaxis_label,
    ymin,
    ymax,
    hide_title,
    selected_dbs,
    show_individual,
):
    if not _databases:
        return _no_data_msg()
    if not selected_dbs:
        return _no_selection_msg()
    fig = _novelty_fig(
        show_mean,
        show_max,
        window,
        theme,
        yaxis_label,
        ymin,
        ymax,
        hide_title,
        selected_dbs,
        show_individual,
    )
    return _wrap_plotly(fig, font_size) if fig is not None else _no_data_msg()


def _make_novelty_page() -> pn.Column:
    return pn.Column(
        pn.pane.Markdown("## Novelty over Generations"),
        pn.pane.Markdown(
            "**N(x, archive)** = mean Euclidean distance from individual *x* to every "
            "individual alive in the **N previous generations** (the archive).  \n"
            "Adjust **N** with the slider below, and toggle lines in the sidebar.",
        ),
        sld_nov_window,
        pn.Row(txt_nov_ymin, txt_nov_ymax),
        chk_nov_hide_title,
        pn.bind(
            _novelty_plot,
            show_mean=chk_nov_mean,
            show_max=chk_nov_max,
            window=sld_nov_window,
            theme=sel_colour_theme,
            font_size=sld_font_size,
            yaxis_label=txt_nov_yaxis,
            ymin=txt_nov_ymin,
            ymax=txt_nov_ymax,
            hide_title=chk_nov_hide_title,
            selected_dbs=mc_db_select,
            show_individual=chk_show_individual,
        ),
        _pdf_download_btn(
            lambda: _novelty_fig(
                chk_nov_mean.value,
                chk_nov_max.value,
                sld_nov_window.value,
                sel_colour_theme.value,
                txt_nov_yaxis.value,
                txt_nov_ymin.value,
                txt_nov_ymax.value,
                chk_nov_hide_title.value,
                mc_db_select.value,
                chk_show_individual.value,
            ),
            "novelty.pdf",
        ),
        align="center",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 - Age over time  (reactive)
# ─────────────────────────────────────────────────────────────────────────────
def _age_fig(
    show_mean,
    show_max,
    show_median,
    theme,
    yaxis_label,
    ymin,
    ymax,
    hide_title,
    selected_dbs,
    show_individual,
) -> go.Figure | None:
    if not selected_dbs:
        return None
    colors = _COLOUR_THEMES[theme]
    fig = go.Figure()
    yrange = _parse_yrange(ymin, ymax)

    if show_individual:
        for i, label in enumerate(selected_dbs):
            entry = _databases.get(label)
            if entry is None:
                continue
            s = entry["stats"]
            col = colors[i % len(colors)]
            if show_mean:
                fig.add_trace(
                    go.Scatter(
                        x=s["generations"],
                        y=s["age_mean"],
                        mode="lines",
                        line={"color": col, "width": 2},
                        legendgroup=label,
                        name=f"{label} — Mean",
                    ),
                )
            if show_max:
                fig.add_trace(
                    go.Scatter(
                        x=s["generations"],
                        y=s["age_max"],
                        mode="lines",
                        line={"color": col, "width": 1.5, "dash": "dash"},
                        legendgroup=label,
                        name=f"{label} — Max",
                    ),
                )
            if show_median:
                fig.add_trace(
                    go.Scatter(
                        x=s["generations"],
                        y=s["age_median"],
                        mode="lines",
                        line={"color": col, "width": 1.5, "dash": "dot"},
                        legendgroup=label,
                        name=f"{label} — Median",
                    ),
                )
    else:
        s = _aggregate_stats(selected_dbs)
        if s is None:
            return None
        if show_mean:
            fig.add_trace(
                go.Scatter(
                    x=s["generations"],
                    y=s["age_mean"],
                    mode="lines",
                    line={"color": colors[0], "width": 2},
                    name="Mean age",
                ),
            )
        if show_max:
            fig.add_trace(
                go.Scatter(
                    x=s["generations"],
                    y=s["age_max"],
                    mode="lines",
                    line={"color": colors[1], "width": 1.5, "dash": "dash"},
                    name="Max age",
                ),
            )
        if show_median:
            fig.add_trace(
                go.Scatter(
                    x=s["generations"],
                    y=s["age_median"],
                    mode="lines",
                    line={"color": colors[2], "width": 1.5, "dash": "dot"},
                    name="Median age",
                ),
            )

    fig.update_layout(
        title="" if hide_title else "Individual lifetime over generations",
        xaxis_title="Generation",
        yaxis_title=yaxis_label,
        yaxis={"range": list(yrange) if yrange else None},
    )
    return fig


def _age_plot(
    show_mean,
    show_max,
    show_median,
    theme,
    font_size,
    yaxis_label,
    ymin,
    ymax,
    hide_title,
    selected_dbs,
    show_individual,
):
    if not _databases:
        return _no_data_msg()
    if not selected_dbs:
        return _no_selection_msg()
    fig = _age_fig(
        show_mean,
        show_max,
        show_median,
        theme,
        yaxis_label,
        ymin,
        ymax,
        hide_title,
        selected_dbs,
        show_individual,
    )
    return _wrap_plotly(fig, font_size) if fig is not None else _no_data_msg()


def _make_age_page() -> pn.Column:
    return pn.Column(
        pn.pane.Markdown("## Age over Generations"),
        pn.pane.Markdown(
            "**L(x)** = number of generations since the birth of individual *x* "
            "(i.e. current generation minus time_of_birth).  \n"
            "Toggle the **Age** options in the sidebar.",
        ),
        pn.Row(txt_age_ymin, txt_age_ymax),
        chk_age_hide_title,
        pn.bind(
            _age_plot,
            show_mean=chk_age_mean,
            show_max=chk_age_max,
            show_median=chk_age_median,
            theme=sel_colour_theme,
            font_size=sld_font_size,
            yaxis_label=txt_age_yaxis,
            ymin=txt_age_ymin,
            ymax=txt_age_ymax,
            hide_title=chk_age_hide_title,
            selected_dbs=mc_db_select,
            show_individual=chk_show_individual,
        ),
        _pdf_download_btn(
            lambda: _age_fig(
                chk_age_mean.value,
                chk_age_max.value,
                chk_age_median.value,
                sel_colour_theme.value,
                txt_age_yaxis.value,
                txt_age_ymin.value,
                txt_age_ymax.value,
                chk_age_hide_title.value,
                mc_db_select.value,
                chk_show_individual.value,
            ),
            "age.pdf",
        ),
        align="center",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Navigation buttons
# ─────────────────────────────────────────────────────────────────────────────
_BTN = {"styles": {"width": "100%"}}

btn_load = pn.widgets.Button(
    name="Load Database", button_type="warning", icon="database", **_BTN,
)
btn_fit = pn.widgets.Button(
    name="Fitness", button_type="warning", icon="chart-line", **_BTN,
)
btn_div = pn.widgets.Button(
    name="Diversity", button_type="warning", icon="arrows-split-2", **_BTN,
)
btn_nov = pn.widgets.Button(
    name="Novelty", button_type="warning", icon="sparkles", **_BTN,
)
btn_age = pn.widgets.Button(
    name="Age", button_type="warning", icon="clock", **_BTN,
)

# ─────────────────────────────────────────────────────────────────────────────
# Main area & routing
# ─────────────────────────────────────────────────────────────────────────────
main_area = pn.Column(_make_load_page(), styles={"width": "100%"})


def _show(key: str) -> None:
    main_area.clear()
    if key == "load":
        main_area.append(_make_load_page())
    elif key == "fitness":
        main_area.append(_make_fitness_page())
    elif key == "diversity":
        main_area.append(_make_diversity_page())
    elif key == "novelty":
        main_area.append(_make_novelty_page())
    elif key == "age":
        main_area.append(_make_age_page())


btn_load.on_click(lambda _: _show("load"))
btn_fit.on_click(lambda _: _show("fitness"))
btn_div.on_click(lambda _: _show("diversity"))
btn_nov.on_click(lambda _: _show("novelty"))
btn_age.on_click(lambda _: _show("age"))

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar  (navigation + per-plot controls, always visible)
# ─────────────────────────────────────────────────────────────────────────────
sidebar = pn.Column(
    pn.pane.Markdown("## Pages"),
    btn_load,
    btn_fit,
    btn_div,
    btn_nov,
    btn_age,
    pn.layout.Divider(),
    pn.pane.Markdown("### Databases"),
    mc_db_select,
    chk_show_individual,
    pn.layout.Divider(),
    pn.pane.Markdown("### Fitness"),
    chk_fit_mean,
    chk_fit_std,
    chk_fit_max,
    chk_fit_min,
    txt_fit_yaxis,
    # pn.Row(txt_fit_ymin, txt_fit_ymax),  from testing
    # chk_fit_hide_title,
    pn.layout.Divider(),
    pn.pane.Markdown("### Diversity"),
    txt_div_yaxis,
    # pn.Row(txt_div_ymin, txt_div_ymax), from testing
    # chk_div_hide_title,
    pn.layout.Divider(),
    pn.pane.Markdown("### Novelty"),
    chk_nov_mean,
    chk_nov_max,
    sld_nov_window,
    txt_nov_yaxis,
    # pn.Row(txt_nov_ymin, txt_nov_ymax), from testing
    # chk_nov_hide_title,
    pn.layout.Divider(),
    pn.pane.Markdown("### Age"),
    chk_age_mean,
    chk_age_max,
    chk_age_median,
    txt_age_yaxis,
    # pn.Row(txt_age_ymin, txt_age_ymax), from testing
    # chk_age_hide_title,
    pn.layout.Divider(),
    pn.pane.Markdown("### Plot Style"),
    sel_colour_theme,
    sld_font_size,
    sld_fig_width,
    styles={"width": "100%", "padding": "15px"},
)

# ─────────────────────────────────────────────────────────────────────────────
# App template  (light mode - no DarkTheme kwarg)
# ─────────────────────────────────────────────────────────────────────────────
template = pn.template.BootstrapTemplate(
    title="ARIEL - Evolution Dashboard",
    sidebar=[sidebar],
    main=[main_area],
    header_background="#1f2e45",
    sidebar_width=400,
    busy_indicator=None,
)

template.servable()
