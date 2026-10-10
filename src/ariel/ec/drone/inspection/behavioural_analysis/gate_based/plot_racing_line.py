import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import PowerNorm
import matplotlib.pyplot as plt
import torch

from ariel.ec.drone.inspection.behavioural_analysis.extract_simulation_data import extract_simulation_data


def plot_racing_line_speed(ind_speed, ind_positions_2d, save_dir, save_name=None, 
                            vmin=None, vmax=None, colormap='viridis', 
                            cbar_label="Speed", title="Best Individual", 
                            suptitle="Drone Racing Line (Colored by Speed)",
                            width=864, height=700, dpi=200, min_timestep=0, max_timestep=None):
    """
    Plots the racing line for speed for a single individual.

    Parameters:
        ind_speed (numpy.ndarray): Array of speeds for the individual.
        ind_positions_2d (numpy.ndarray): 2D positions of the individual.
        save_dir (str): Directory to save the plot.
    """
    # Trim the data to the specified time range
    if max_timestep is None:
        max_timestep = ind_positions_2d.shape[0]-1
    ind_positions_2d = ind_positions_2d[min_timestep:max_timestep]
    ind_speed = ind_speed[min_timestep:max_timestep]
    ind_speed = ind_speed[min_timestep:max_timestep]  # Speed is one less than positions
    ind_positions_2d = ind_positions_2d[min_timestep:max_timestep]  # Positions are one less than speed

    points = ind_positions_2d.reshape(-1, 1, 2)
    ind_segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Normalize speed
    if vmin is None:
        vmin = ind_speed.min()
    if vmax is None:
        vmax = ind_speed.max()
    norm_speed = PowerNorm(gamma=3.0, vmin=vmin, vmax=vmax)

    # Create LineCollection for speed
    ind_lc_speed = LineCollection(ind_segments, cmap=colormap, norm=norm_speed)
    ind_lc_speed.set_array(ind_speed[:-1])
    ind_lc_speed.set_linewidth(2)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi))

    # Add LineCollection to the plot
    ax.add_collection(ind_lc_speed)
    ax.set_xlim(ind_positions_2d[:, 0].min(), ind_positions_2d[:, 0].max())
    ax.set_ylim(ind_positions_2d[:, 1].min(), ind_positions_2d[:, 1].max())
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Add colorbar
    cbar = fig.colorbar(ind_lc_speed, ax=ax, orientation='horizontal', fraction=0.02, pad=0.1)
    cbar.set_label(cbar_label)

    # Add title
    ax.text(0.5, -0.2, title, ha='center', va='center', transform=ax.transAxes)
    fig.suptitle(suptitle)

    # Adjust layout and save the figure
    fig.tight_layout()
    if save_name is None:
        save_name = "racing_line_speed.png"
    fig.savefig(f"{save_dir}{save_name}", dpi=dpi)
    plt.close(fig)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import PowerNorm

def plot_racing_line_speed_segments(ind_speed, ind_positions_2d, save_dir, save_name=None, 
                                    vmin=None, vmax=None, colormap='viridis', 
                                    cbar_label="Speed", suptitle="Drone Racing Line Segments (Colored by Speed)",
                                    width=864, height=700, dpi=200, min_timestep=0, max_timestep=None):
    """
    Plots the racing line for speed, splitting the track into 4 segments and displaying them in a column.

    Parameters:
        ind_speed (numpy.ndarray): Array of speeds for the individual.
        ind_positions_2d (numpy.ndarray): 2D positions of the individual.
        save_dir (str): Directory to save the plot.
    """
    # Trim the data to the specified time range
    if max_timestep is None:
        max_timestep = ind_positions_2d.shape[0]-12
    ind_positions_2d = ind_positions_2d[min_timestep:max_timestep]
    ind_speed = ind_speed[min_timestep:max_timestep]

    # Split data into 4 segments
    num_points = ind_positions_2d.shape[0]
    segment_size = num_points // 4
    segments_positions = [ind_positions_2d[i * segment_size:(i + 1) * segment_size] for i in range(4)]
    segments_speed = [ind_speed[i * segment_size:(i + 1) * segment_size] for i in range(4)]

    # Normalize speed
    if vmin is None:
        vmin = ind_speed.min()
    if vmax is None:
        vmax = ind_speed.max()
    norm_speed = PowerNorm(gamma=3.0, vmin=vmin, vmax=vmax)

    # Create figure and subplots
    fig, axes = plt.subplots(4, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(hspace=0.5)

    for i, (positions, speeds, ax) in enumerate(zip(segments_positions, segments_speed, axes)):
        # Create segments for LineCollection
        points = positions.reshape(-1, 1, 2)
        ind_segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create LineCollection for speed
        ind_lc_speed = LineCollection(ind_segments, cmap=colormap, norm=norm_speed)
        ind_lc_speed.set_array(speeds[:-1])
        ind_lc_speed.set_linewidth(2)

        # Add LineCollection to the plot
        ax.add_collection(ind_lc_speed)
        ax.set_xlim(positions[:, 0].min(), positions[:, 0].max())
        ax.set_ylim(positions[:, 1].min(), positions[:, 1].max())
        ax.set_aspect('equal')
        ax.grid(True)
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_title(f"Segment {i + 1}")

        # Add colorbar
        # cbar = fig.colorbar(ind_lc_speed, ax=ax, orientation='horizontal', fraction=0.02, pad=0.1)
        # cbar.set_label(cbar_label)

    # Add overall title
    fig.suptitle(suptitle)

    # Adjust layout and save the figure
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_name is None:
        save_name = "racing_line_speed_segments.png"
    fig.savefig(f"{save_dir}{save_name}", dpi=dpi)
    plt.close(fig)

if __name__ == "__main__":
    # Example usage
    save_dir = "./plots/example_comparison/"
    ind_dir =  "/home/jed/workspaces/airevolve/data_backup/asym_slalom/asym_slalom4evo_logs_20250320_095329/gen40/ind964/"
    policy_file = ind_dir + "/policy.zip"
    body = ind_dir + "/individual.npy"
    individual = np.load(body)
    gate_cfg = "slalom"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data = extract_simulation_data(individual, policy_file, gate_cfg, device)
    speed = np.linalg.norm(data["velocities"], axis=1)
    angular_speed = np.linalg.norm(data["angular_velocities"], axis=1)
    positions_2d = data["positions"][:, :2]

    plot_racing_line_speed_segments(
        ind_speed=speed,
        ind_positions_2d=positions_2d,
        save_dir=save_dir,
        save_name="racing_line_speed.png",
        vmin=0,  # Minimum speed for normalization
        vmax=13,  # Maximum speed for normalization
        colormap="plasma",  # Colormap for the plot
        cbar_label="Speed (m/s)",  # Label for the colorbar
        # title="Example Racing Line",  # Title of the plot
        suptitle="Drone Racing Line Example",
        min_timestep=0,
        max_timestep=None,  # Maximum timestep for the plot
)