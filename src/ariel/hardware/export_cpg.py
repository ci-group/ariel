"""Workstation utility: export a trained SimpleCPG to a hardware-ready `.npz`.

Run this on the machine where torch is installed, then copy the resulting
`.npz` to the robot.

Example
-------
    from ariel.simulation.controllers.simple_cpg import SimpleCPG, create_fully_connected_adjacency
    from ariel.hardware.export_cpg import export_simple_cpg

    cpg = SimpleCPG(create_fully_connected_adjacency(8))
    cpg.set_flat_params(trained_params)

    export_simple_cpg(cpg, "spider_cpg.npz")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_simple_cpg(cpg, path: str | Path) -> None:
    """Serialize a trained :class:`SimpleCPG` to a numpy `.npz` file.

    The output can be loaded on the robot (no torch required) via
    :meth:`ariel.hardware.cpg_inference.SimpleCPGInference.load`.

    Parameters
    ----------
    cpg : SimpleCPG
        A trained SimpleCPG instance (torch).
    path : str | Path
        Destination file path.  The `.npz` extension is added automatically
        by :func:`numpy.savez` if omitted.
    """
    import json

    adjacency_json = json.dumps(
        {str(k): v for k, v in cpg.adjacency_dict.items()}
    )
    hb = cpg.hard_bounds
    np.savez(
        Path(path),
        phase=cpg.phase.detach().cpu().numpy(),
        w=cpg.w.detach().cpu().numpy(),
        amplitudes=cpg.amplitudes.detach().cpu().numpy(),
        ha=cpg.ha.detach().cpu().numpy(),
        b=cpg.b.detach().cpu().numpy(),
        x0=cpg.x.detach().cpu().numpy(),
        y0=cpg.y.detach().cpu().numpy(),
        mu=np.array(cpg.mu, dtype=np.float32),
        dt=np.array(cpg.dt, dtype=np.float32),
        hard_bounds=np.array(hb, dtype=np.float32)
        if hb is not None
        else np.array([np.nan, np.nan], dtype=np.float32),
        has_hard_bounds=np.array(hb is not None),
        adjacency_json=np.array(adjacency_json),
    )
    print(f"[export_cpg] Saved to {Path(path)}")
