"""Visualize the best evolved drone from NEAT evolution.

Loads the best individual from a NEAT evolution run and displays it in:
  • 3D isometric view (matplotlib)
  • 2D top-down view (matplotlib)
  • Blueprint 4-panel view (matplotlib)
  • Interactive 3D simulator (optional)

Run:
    python examples/d_drones/5_visualize_best_drone.py
    python examples/d_drones/5_visualize_best_drone.py --no-show
    python examples/d_drones/5_visualize_best_drone.py --simulate
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from rich.console import Console

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from ariel.ec.drone.genome_handlers.spherical_angular_genome_handler import (
    SphericalAngularDroneGenomeHandler,
)
from ariel.ec.drone.inspection.drone_visualizer import (
    DroneVisualizer,
    VisualizationConfig,
)

console = Console()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Visualize the best evolved drone from NEAT evolution"
)
parser.add_argument(
    "--run-dir",
    type=str,
    default="__data__/drone_neat_evolution",
    help="Directory containing NEAT evolution results",
)
parser.add_argument(
    "--generation",
    type=int,
    default=None,
    help="Which generation to load from (default: final generation)",
)
parser.add_argument(
    "--individual-id",
    type=str,
    default="0600",
    help="Specific individual ID to visualize (default: 0600)",
)
parser.add_argument(
    "--no-show",
    action="store_true",
    help="Save figures without showing interactive windows",
)
parser.add_argument(
    "--simulate",
    action="store_true",
    help="Also simulate the drone using physics simulator (if available)",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="__data__/visualizations/neat_best_drone",
    help="Output directory for saved figures",
)
args = parser.parse_args()

if args.no_show:
    matplotlib.use("Agg")

RUN_DIR = Path.cwd() / args.run_dir
OUTPUT_DIR = Path.cwd() / args.output_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load best individual
# ---------------------------------------------------------------------------


def find_best_individual() -> tuple[str, str, str, np.ndarray]:
    """Find an individual from evolution run.

    Returns:
        (generation_dir, individual_dir, individual_id, genome)
    """
    if not RUN_DIR.exists():
        console.print(f"[red]Error: {RUN_DIR} not found[/red]")
        raise FileNotFoundError(f"Run directory {RUN_DIR} not found")

    # List all generation directories
    gen_dirs = sorted([d for d in RUN_DIR.iterdir() if d.is_dir() and d.name.startswith("generation_")])
    
    if not gen_dirs:
        console.print(f"[red]Error: No generation directories found in {RUN_DIR}[/red]")
        raise FileNotFoundError(f"No generation directories found in {RUN_DIR}")

    # Use specified generation or last one
    if args.generation is not None:
        gen_name = f"generation_{args.generation:02d}"
        target_gen_dir = RUN_DIR / gen_name
        if not target_gen_dir.exists():
            console.print(f"[red]Error: {target_gen_dir} not found[/red]")
            raise FileNotFoundError(f"Generation directory {target_gen_dir} not found")
        gen_dir = target_gen_dir
    else:
        gen_dir = gen_dirs[-1]  # Use final generation

    gen_name = gen_dir.name
    console.log(f"Searching {gen_name}...")
    
    # Determine which individual to load
    target_ind_id = args.individual_id
    target_ind_dir = gen_dir / f"individual_{target_ind_id}"
    
    if not target_ind_dir.exists():
        # If that doesn't exist, list available individuals
        ind_dirs = sorted([d for d in gen_dir.iterdir() if d.is_dir() and d.name.startswith("individual_")])
        if ind_dirs:
            # Use the first one
            target_ind_dir = ind_dirs[0]
            target_ind_id = target_ind_dir.name.split("_")[1]
            console.log(f"[yellow]ID {args.individual_id} not found, using {target_ind_id}[/yellow]")
        else:
            raise FileNotFoundError(f"No individuals found in {gen_dir}")
    
    genome_path = target_ind_dir / "genome.npy"
    
    if not genome_path.exists():
        raise FileNotFoundError(f"Genome not found at {genome_path}")
    
    # Load genome (allow_pickle=True needed for object arrays)
    genome_obj = np.load(genome_path, allow_pickle=True)
    
    # If it's wrapped in a 0-d array, unwrap it
    if genome_obj.shape == ():
        genome = genome_obj.item()
    else:
        genome = genome_obj
    
    console.log(f"[green]Loaded individual:[/green]")
    console.log(f"  Generation: {gen_name}")
    console.log(f"  Individual ID: {target_ind_id}")
    console.log(f"  Genome path: {genome_path}")
    if hasattr(genome, 'arms'):
        console.log(f"  Genome type: SphericalNeatGenome")
        console.log(f"  Arms shape: {genome.arms.shape}")
    else:
        console.log(f"  Genome shape: {genome.shape if hasattr(genome, 'shape') else 'N/A'}")
    
    return gen_name, target_ind_dir.name, target_ind_id, genome


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

console.rule("[bold blue]Drone Visualization")

gen_name, ind_dir, ind_id, genome = find_best_individual()

# Extract phenotype (valid arms only)
# The genome is a SphericalNeatGenome object, extract the arms
if hasattr(genome, 'arms'):
    # It's a namedtuple or object with arms attribute
    arms_data = genome.arms
else:
    # It's already a numpy array
    arms_data = genome

valid_mask = ~np.isnan(arms_data[:, 0])
phenotype = arms_data[valid_mask]

console.log(f"[cyan]Phenotype shape: {phenotype.shape}[/cyan]")
console.log(f"  Arms: {phenotype.shape[0]}")
console.log(f"  Parameters per arm: {phenotype.shape[1]}")

# Create visualizer
visualizer = DroneVisualizer()

# Visualization config
config = VisualizationConfig(
    circle_radius=0.0254,
    scale_factor=1.2,
    elevation=30,
    azimuth=30,
)

# 3D Isometric View
# ---------------------------------------------------------------------------

console.log("\n[bold cyan]Creating 3D isometric view...[/bold cyan]")
try:
    fig_3d, ax_3d = visualizer.plot_3d(
        phenotype,
        title=f"Best Drone (Gen {gen_name}, ID {ind_id})",
        generation=int(gen_name.split("_")[1]),
    )
    fig_3d.tight_layout()
    fig_3d.savefig(OUTPUT_DIR / "01_3d_isometric.png", dpi=150)
    console.log(f"  ✓ Saved to {OUTPUT_DIR / '01_3d_isometric.png'}")
except Exception as e:
    console.log(f"  [red]Error creating 3D plot: {e}[/red]")

# ---------------------------------------------------------------------------
# 2D Top-Down View
# ---------------------------------------------------------------------------

console.log("[bold cyan]Creating 2D top-down view...[/bold cyan]")
try:
    fig_2d, ax_2d = visualizer.plot_2d(
        phenotype,
        title=f"Best Drone - Top View (Gen {gen_name})",
        xlim=(-0.5, 0.5),
        ylim=(-0.5, 0.5),
    )
    fig_2d.tight_layout()
    fig_2d.savefig(OUTPUT_DIR / "02_2d_topdown.png", dpi=150)
    console.log(f"  ✓ Saved to {OUTPUT_DIR / '02_2d_topdown.png'}")
except Exception as e:
    console.log(f"  [red]Error creating 2D plot: {e}[/red]")

# ---------------------------------------------------------------------------
# Blueprint (4-panel) View
# ---------------------------------------------------------------------------

console.log("[bold cyan]Creating blueprint 4-panel view...[/bold cyan]")
try:
    fig_bp, axes_bp = visualizer.plot_blueprint(
        phenotype,
        title=f"Best Drone Blueprint (Gen {gen_name}, ID {ind_id})",
        figsize=(14, 14),
    )
    fig_bp.tight_layout()
    fig_bp.savefig(OUTPUT_DIR / "03_blueprint.png", dpi=150)
    console.log(f"  ✓ Saved to {OUTPUT_DIR / '03_blueprint.png'}")
except Exception as e:
    console.log(f"  [red]Error creating blueprint plot: {e}[/red]")

# ---------------------------------------------------------------------------
# Simulation (optional)
# ---------------------------------------------------------------------------

if args.simulate:
    console.log("\n[bold cyan]Attempting physics simulation...[/bold cyan]")
    try:
        from ariel.ec.drone.evaluators.hover_fitness import (
            create_drone_simulator,
        )
        import mujoco
        
        # Create simulator for the drone
        console.log("  Creating drone simulator...")
        simulator = create_drone_simulator(phenotype)
        
        console.log("  [green]✓ Simulator created successfully[/green]")
        console.log("  (Use 'python examples/d_drones/3_simulate_lee.py' for interactive simulation)")
        
    except ImportError as e:
        console.log(f"  [yellow]Simulator not available: {e}[/yellow]")
    except Exception as e:
        console.log(f"  [red]Error creating simulator: {e}[/red]")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

console.rule("[bold green]Visualization Complete")
console.log(f"Output directory: {OUTPUT_DIR}")
console.log(f"Files generated:")
console.log(f"  • 01_3d_isometric.png")
console.log(f"  • 02_2d_topdown.png")
console.log(f"  • 03_blueprint.png")

if not args.no_show:
    plt.show()

console.log("[bold]Done![/bold]")
