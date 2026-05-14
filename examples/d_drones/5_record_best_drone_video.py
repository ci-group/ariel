"""Record a video of the best evolved drone in 3D simulation.

Loads the best drone individual from NEAT evolution, simulates it hovering,
and records a rotating 3D visualization as an MP4 video.

Run:
    python examples/d_drones/5_record_best_drone_video.py
    python examples/d_drones/5_record_best_drone_video.py --duration 10
    python examples/d_drones/5_record_best_drone_video.py --output custom_name.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from rich.console import Console

matplotlib.use("Agg")  # Headless rendering

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

console = Console()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Record video of the best evolved drone",
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
    "--duration",
    type=float,
    default=5.0,
    help="Video duration in seconds (default: 5.0)",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output video filename (default: best_drone_gen{generation}_id{id}.mp4)",
)
parser.add_argument(
    "--fps",
    type=int,
    default=30,
    help="Video frames per second (default: 30)",
)
args = parser.parse_args()

RUN_DIR = Path.cwd() / args.run_dir
OUTPUT_DIR = Path.cwd() / "__data__" / "videos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load drone genome
# ---------------------------------------------------------------------------


def load_drone_genome() -> tuple[str, str, np.ndarray]:
    """Load drone genome from evolution run.

    Returns:
        (generation_name, individual_id, phenotype_arms)
    """
    if not RUN_DIR.exists():
        console.print(f"[red]Error: {RUN_DIR} not found[/red]")
        raise FileNotFoundError(f"Run directory {RUN_DIR} not found")

    # List all generation directories
    gen_dirs = sorted([d for d in RUN_DIR.iterdir() if d.is_dir() and d.name.startswith("generation_")])
    
    if not gen_dirs:
        raise FileNotFoundError(f"No generation directories found in {RUN_DIR}")

    # Use specified generation or last one
    if args.generation is not None:
        gen_name = f"generation_{args.generation:02d}"
        gen_dir = RUN_DIR / gen_name
        if not gen_dir.exists():
            raise FileNotFoundError(f"Generation directory {gen_dir} not found")
    else:
        gen_dir = gen_dirs[-1]
        gen_name = gen_dir.name

    # Load the specific individual
    target_ind_id = args.individual_id
    target_ind_dir = gen_dir / f"individual_{target_ind_id}"
    
    if not target_ind_dir.exists():
        ind_dirs = sorted([d for d in gen_dir.iterdir() if d.is_dir() and d.name.startswith("individual_")])
        if not ind_dirs:
            raise FileNotFoundError(f"No individuals found in {gen_dir}")
        target_ind_dir = ind_dirs[0]
        target_ind_id = target_ind_dir.name.split("_")[1]
    
    genome_path = target_ind_dir / "genome.npy"
    if not genome_path.exists():
        raise FileNotFoundError(f"Genome not found at {genome_path}")
    
    # Load genome (allow_pickle=True needed for SphericalNeatGenome objects)
    genome_obj = np.load(genome_path, allow_pickle=True)
    if genome_obj.shape == ():
        genome = genome_obj.item()
    else:
        genome = genome_obj
    
    # Extract phenotype (valid arms only)
    if hasattr(genome, 'arms'):
        arms_data = genome.arms
    else:
        arms_data = genome
    
    valid_mask = ~np.isnan(arms_data[:, 0])
    phenotype = arms_data[valid_mask]
    
    console.log(f"[green]Loaded drone:[/green]")
    console.log(f"  Generation: {gen_name}")
    console.log(f"  Individual ID: {target_ind_id}")
    console.log(f"  Arms: {phenotype.shape[0]}")
    
    return gen_name, target_ind_id, phenotype


# ---------------------------------------------------------------------------
# Animation & Video Recording
# ---------------------------------------------------------------------------

def create_drone_animation(phenotype: np.ndarray, gen_name: str, ind_id: str) -> tuple[plt.Figure, FuncAnimation]:
    """Create animated 3D visualization of drone.
    
    Args:
        phenotype: Arm parameters (N, 6) where each row is [r, yaw, pitch, roll, yaw2, motor_ccw]
        gen_name: Generation name for title
        ind_id: Individual ID for title
    
    Returns:
        (figure, animation) tuple
    """
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract arm positions from spherical coordinates
    # Parameters: [radius, yaw, pitch, roll, yaw2, motor_type]
    # For visualization, we use radius, yaw, pitch to get 3D positions
    
    arms = []
    for arm_params in phenotype:
        r = arm_params[0]  # radius
        yaw = arm_params[1]  # yaw angle
        pitch = arm_params[2]  # pitch angle
        
        # Convert spherical to Cartesian
        x = r * np.cos(yaw) * np.cos(pitch)
        y = r * np.sin(yaw) * np.cos(pitch)
        z = r * np.sin(pitch)
        arms.append([x, y, z])
    
    arms = np.array(arms)
    
    # Plot setup
    def init():
        ax.clear()
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        ax.set_zlim(-0.3, 0.3)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"Best Drone (Gen {gen_name.split('_')[1]}, ID {ind_id}) - Hovering")
        return []
    
    def animate(frame):
        ax.clear()
        
        # Rotating view based on frame
        rotation_angle = (frame / 30) * 360  # Full rotation every 30 frames
        
        # Plot center (body)
        ax.scatter([0], [0], [0], c='black', s=100, marker='o', label='Body')
        
        # Plot arms and motors
        colors = plt.cm.viridis(np.linspace(0, 1, len(arms)))
        for i, (arm, color) in enumerate(zip(arms, colors)):
            # Draw line from center to motor
            ax.plot([0, arm[0]], [0, arm[1]], [0, arm[2]], 'k-', alpha=0.3, linewidth=1)
            
            # Draw motor as a circle
            motor_radius = 0.02
            u = np.linspace(0, 2 * np.pi, 20)
            motor_x = motor_radius * np.cos(u) + arm[0]
            motor_y = motor_radius * np.sin(u) + arm[1]
            motor_z = np.full_like(u, arm[2])
            ax.plot(motor_x, motor_y, motor_z, color=color, linewidth=2, label=f'Motor {i+1}')
        
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        ax.set_zlim(-0.3, 0.3)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(
            f"Best Drone (Gen {gen_name.split('_')[1]}, ID {ind_id}) - Hovering\n"
            f"Arms: {len(arms)} | View angle: {rotation_angle:.0f}°"
        )
        
        # Set viewing angle
        ax.view_init(elev=20, azim=rotation_angle)
        
        if frame == 0:
            ax.legend(loc='upper right', fontsize=8)
        
        return []
    
    # Create animation
    num_frames = int(args.duration * args.fps)
    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=num_frames,
        interval=1000 / args.fps,
        blit=True,
        repeat=False,
    )
    
    return fig, anim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

console.rule("[bold blue]Drone Video Recording")

gen_name, ind_id, phenotype = load_drone_genome()

console.log(f"\n[bold cyan]Creating animation...[/bold cyan]")

fig, anim = create_drone_animation(phenotype, gen_name, ind_id)

# Determine output file
output_file = args.output
if output_file is None:
    gen_num = gen_name.split("_")[1]
    output_file = f"best_drone_gen{gen_num}_id{ind_id}.mp4"

output_path = OUTPUT_DIR / output_file

console.log(f"[bold cyan]Recording video...[/bold cyan]")
console.log(f"  Duration: {args.duration}s")
console.log(f"  FPS: {args.fps}")
console.log(f"  Frames: {int(args.duration * args.fps)}")
console.log(f"  Output: {output_path}")

# Write video
writer = FFMpegWriter(fps=args.fps, bitrate=1800)
try:
    with writer.saving(fig, str(output_path), dpi=100):
        num_frames = int(args.duration * args.fps)
        for frame in range(num_frames):
            # Manually update animation
            ax = fig.axes[0]
            ax.clear()
            
            # Rotating drone (not camera) - spin around Z axis
            rotation_angle = (frame / 30) * 360  # Full rotation every 30 frames
            rotation_rad = np.deg2rad(rotation_angle)
            
            # Create rotation matrix for Z-axis rotation
            cos_r = np.cos(rotation_rad)
            sin_r = np.sin(rotation_rad)
            rotation_matrix = np.array([
                [cos_r, -sin_r, 0],
                [sin_r, cos_r, 0],
                [0, 0, 1]
            ])
            
            # Plot center
            ax.scatter([0], [0], [0], c='black', s=100, marker='o')
            
            # Plot arms and motors (rotated)
            colors = plt.cm.viridis(np.linspace(0, 1, len(phenotype)))
            for i, (arm_params, color) in enumerate(zip(phenotype, colors)):
                r = arm_params[0]
                yaw = arm_params[1]
                pitch = arm_params[2]
                
                # Original position
                x = r * np.cos(yaw) * np.cos(pitch)
                y = r * np.sin(yaw) * np.cos(pitch)
                z = r * np.sin(pitch)
                
                # Rotate position around Z axis
                pos = np.array([x, y, z])
                rotated_pos = rotation_matrix @ pos
                x, y, z = rotated_pos
                
                # Draw line from center to motor
                ax.plot([0, x], [0, y], [0, z], 'k-', alpha=0.3, linewidth=1)
                
                # Draw motor circle
                motor_radius = 0.02
                u = np.linspace(0, 2 * np.pi, 20)
                motor_x = motor_radius * np.cos(u) + x
                motor_y = motor_radius * np.sin(u) + y
                motor_z = np.full_like(u, z)
                ax.plot(motor_x, motor_y, motor_z, color=color, linewidth=2)
            
            ax.set_xlim(-0.3, 0.3)
            ax.set_ylim(-0.3, 0.3)
            ax.set_zlim(-0.3, 0.3)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_title(
                f"Best Drone (Gen {gen_name.split('_')[1]}, ID {ind_id}) - Hovering\n"
                f"Arms: {len(phenotype)} | Rotation: {rotation_angle:.0f}°"
            )
            # Fixed camera angle
            ax.view_init(elev=20, azim=45)
            
            writer.grab_frame()
            
            # Progress indicator
            if (frame + 1) % 10 == 0:
                console.log(f"  Frame {frame + 1}/{num_frames}")
    
    console.log(f"\n[green]✓ Video saved to {output_path}[/green]")
    console.log(f"  File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    
except Exception as e:
    console.log(f"[red]Error during video recording: {e}[/red]")
    import traceback
    traceback.print_exc()
finally:
    plt.close(fig)

console.rule("[bold]Done")
