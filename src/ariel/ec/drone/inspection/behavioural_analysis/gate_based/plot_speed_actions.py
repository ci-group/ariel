import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

import os
import torch

from ariel.ec.drone.inspection.behavioural_analysis.gate_based.calculate_stats import calculate_stats

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_normalized_speeds(speeds, gate_passes, gate_start, gate_end, 
                           title="Normalized Speeds", xlabel="Time (Normalized Between Gates)", ylabel="Speed",
                           fontsize=12, font_size_x_label=12, font_size_y_label=12,
                           font_size_ticks=10, color_palette=None, alphas=None, save_as="normalized_speeds.png", 
                           show_grid=True, grid_linestyle='--', grid_linewidth=0.5, grid_alpha=0.7,
                           show_legend=True, legend_labels=None, legend_fontsize=12, legend_loc='lower right'):
    """
    Plots normalized speeds for each drone, considering gate passes.

    Parameters:
    - speeds: numpy array of shape (num_drones, num_timesteps), speeds of drones
    - gate_passes: numpy array of shape (num_drones, num_timesteps), gate pass information for drones
    - gate_start: starting gate index
    - gate_end: ending gate index
    - title: title of the plot
    - xlabel: label for the x-axis
    - ylabel: label for the y-axis
    - fontsize: font size for the title
    - font_size_x_label: font size for the x-axis label
    - font_size_y_label: font size for the y-axis label
    - font_size_ticks: font size for the x and y ticks
    - color_palette: list of colors for each drone's speed plot
    - save_as: file name to save the plot
    - show_grid: whether to show grid lines on the plot
    - grid_linestyle: linestyle for grid
    - grid_linewidth: linewidth for grid
    - grid_alpha: transparency for grid
    - show_legend: whether to display a legend
    - legend_labels: list of labels for the legend
    """

    # Default color palette if none is provided
    if color_palette is None:
        color_palette = ["red", "green", "blue", "orange", "yellow"]
    if alphas is None:
        alphas = [1.0] * len(color_palette)
    num_drones = speeds.shape[0]
    
    max_num_gate_passes = np.min(np.sum(gate_passes, axis=1))
    print(f"Max number of gate passes: {max_num_gate_passes}")

    # Reduce all speeds to the same number of gate passes
    new_speeds = []
    new_gate_passes = []
    for drone_num in range(num_drones):
        cs = np.cumsum(gate_passes[drone_num])
        when_max_gate_passes = np.where(cs == max_num_gate_passes)[0][0]
        drone_speeds = speeds[drone_num][:when_max_gate_passes]
        drone_gate_passes = gate_passes[drone_num][:when_max_gate_passes]

        new_speeds.append(drone_speeds)
        new_gate_passes.append(drone_gate_passes)

    drone_timesteps_per_segment = []
    drone_speeds_per_segment = []
    for drone_num in range(num_drones):
        drone_speeds = new_speeds[drone_num]
        drone_gate_passes = new_gate_passes[drone_num]

        segments_ends = np.where(drone_gate_passes)[0]
        segments_starts = np.concatenate(([0], segments_ends[:-1] + 1))
        speed_in_segments = [drone_speeds[segments_starts[i]:segments_ends[i]] for i in range(len(segments_starts))]
        drone_speeds_per_segment.append(speed_in_segments)

        timestep_per_segment = []
        for i, segment in enumerate(speed_in_segments):
            timestep_per_segment.append(len(segment))
        drone_timesteps_per_segment.append(timestep_per_segment)
    
    drone_timesteps_per_segment = np.array(drone_timesteps_per_segment)
    max_length_per_segment = np.max(drone_timesteps_per_segment, axis=0)

    fig, ax = plt.subplots(1)
    viable_gates = np.arange(gate_start, gate_end)
    seg_prev_ends = []
    
    if gate_start != 0:
        ax.axvline(x=gate_start, linestyle='dotted', color='purple', alpha=0.75)
    for segment_num in viable_gates:
        drone_prev_ends = []
        for drone_num in range(num_drones):
            drone_speeds_in_segment = drone_speeds_per_segment[drone_num][segment_num]
            mx_ts = drone_timesteps_per_segment[drone_num][segment_num]
            drone_ts_in_segment = np.arange(mx_ts)
            
            timestep_norm = segment_num + (drone_ts_in_segment / mx_ts)
            
            # Handle previous segment ends
            if segment_num != viable_gates[0]:
                timestep_norm = np.insert(timestep_norm, 0, seg_prev_ends[segment_num - viable_gates[0] - 1][drone_num][0])
                drone_speeds_in_segment = np.insert(drone_speeds_in_segment, 0, seg_prev_ends[segment_num - viable_gates[0] - 1][drone_num][1])
            
            ax.plot(timestep_norm, drone_speeds_in_segment, color=color_palette[drone_num], alpha=alphas[drone_num])

            drone_prev_ends.append([timestep_norm[-1], drone_speeds_in_segment[-1]])
        
        seg_prev_ends.append(drone_prev_ends)
        ax.axvline(x=segment_num + 1, linestyle='dotted', color='purple', alpha=0.75)

    # Set plot labels and title
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=font_size_x_label)
    ax.set_ylabel(ylabel, fontsize=font_size_y_label)

    # Customize grid
    if show_grid:
        ax.grid(True, which='both', linestyle=grid_linestyle, linewidth=grid_linewidth, alpha=grid_alpha)

    # Set tick label font size
    ax.tick_params(axis='x', labelsize=font_size_ticks)
    ax.tick_params(axis='y', labelsize=font_size_ticks)

    # Show legend if requested
    legend_labels.append("Gate Passage")
    if show_legend:
        ax.legend(legend_labels, fontsize=legend_fontsize, loc=legend_loc)

    # Save the plot
    fig.savefig(save_as, bbox_inches="tight")



def plot_speed(timesteps, speed, gate_passes, title, save_path, 
               animate=False, ax=None, fps=100, width=864, 
               height=700, dpi=200, gate_lines=False, 
               gate_labels=False, gate_label_ylevel=12.0, 
               fontsize=7, pad=0.05, offset_val=0.5, 
               gate_line_alpha=0.5, alpha=1.0, color='orange'):
    """Plot speed over time with optional animation and support for overlapping plots."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi))
        new_fig = True
    else:
        new_fig = False

    ax.plot(timesteps, speed, label='Speed', color=color, alpha=alpha)
    gate_indices = np.where(gate_passes)[0]

    if gate_lines:
        for i, idx in enumerate(gate_indices):
            ax.axvline(x=idx, color=color, linestyle='--', label='Gate Passage' if i == 0 else "", alpha=gate_line_alpha)
            if gate_labels:
                offset = offset_val if i % 2 == 0 else -offset_val
                yi = speed[idx]
                ax.plot([idx, idx], [yi, gate_label_ylevel+offset], linestyle='dotted', color='gray', linewidth=1)
                ax.text(idx, gate_label_ylevel+offset, str(i), fontsize=fontsize, ha='center', va='center', color='black',
                        bbox=dict(boxstyle=f'circle,pad={pad}', fc=color, alpha=0.5))

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Speed")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    if animate and new_fig:
        fig.subplots_adjust(bottom=0.2) 
        vline = ax.axvline(timesteps[0], color='red', linestyle='--', linewidth=2)

        def update(frame):
            vline.set_xdata([timesteps[frame]])
            return vline,

        ani = FuncAnimation(fig, update, frames=len(timesteps), interval=1000 / fps, blit=True)
        writer = FFMpegWriter(fps=fps)
        ani.save(save_path.replace(".png", ".mp4"), writer=writer, dpi=dpi)
        plt.close()
    elif new_fig:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
def plot_angular_speed(timesteps, angular_speed, gate_passes, title, save_path, 
                       animate=False, fig=None, ax=None, fps=100, width=864, height=700, 
                       dpi=200, gate_lines=False, gate_labels=False, 
                       gate_label_ylevel=11.0, fontsize=7, pad=0.05, 
                       offset_val=0.5, gate_line_alpha=0.5, alpha=1.0, color='blue'):
    """Plot angular speed over time with optional animation and support for overlapping plots."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi))
        new_fig = True
    else:
        new_fig = False

    ax.plot(timesteps, angular_speed, label='Angular Speed', color=color, alpha=alpha)
    gate_indices = np.where(gate_passes)[0]

    if gate_lines:
        for i, idx in enumerate(gate_indices):
            ax.axvline(x=idx, color=color, linestyle='--', label='Gate Passage' if i == 0 else "", alpha=gate_line_alpha)
            if gate_labels:
                offset = offset_val if i % 2 == 0 else -offset_val
                yi = angular_speed[idx]
                ax.plot([idx, idx], [yi, gate_label_ylevel+offset], linestyle='dotted', color='gray', linewidth=1)
                ax.text(idx, gate_label_ylevel+offset, str(i), fontsize=fontsize, ha='center', va='center', color='black',
                        bbox=dict(boxstyle=f'circle,pad={pad}', fc=color, alpha=0.5))

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Angular Speed")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    if animate and new_fig:
        fig.subplots_adjust(bottom=0.2) 
        vline = ax.axvline(timesteps[0], color='red', linestyle='--', linewidth=2)

        def update(frame):
            vline.set_xdata([timesteps[frame]])
            return vline,

        ani = FuncAnimation(fig, update, frames=len(timesteps), interval=1000 / fps, blit=True)
        writer = FFMpegWriter(fps=fps)
        ani.save(save_path.replace(".png", ".mp4"), writer=writer, dpi=dpi)
        plt.close()
    elif new_fig:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

def plot_actions(timesteps, actions, gate_passes, motor_colors, save_path, animate=False, fps=100, width=864, height=1400, dpi=200):
    """Plot motor actions over time with optional animation."""
    nactions = actions.shape[1]
    fig, axs = plt.subplots(nactions, 1, figsize=(width / dpi, height / dpi), sharex=True)

    for i in range(nactions):
        axs[i].plot(timesteps, actions[:, i], label=f'Motor {i+1}', color=motor_colors[i])
        gate_indices = np.where(gate_passes)[0]
        for idx in gate_indices:
            axs[i].axvline(x=idx, color='red', linestyle='--', label='Gate Passage' if idx == gate_indices[0] else "", alpha=0.5)
        axs[i].set_ylabel(f'Motor {i+1}')
        axs[i].set_ylim(-0.1, 1.1)
        axs[i].grid(True)

    axs[-1].set_xlabel("Time Step")
    fig.suptitle("Motor Actions Over Time with Gate Passages")
    fig.tight_layout()

    if animate:
        vlines = [axs[i].axvline(timesteps[0], color='red', linestyle='--', linewidth=2) for i in range(nactions)]

        def update(frame):
            for vline in vlines:
                vline.set_xdata([timesteps[frame]])
            return vlines

        ani = FuncAnimation(fig, update, frames=len(timesteps), interval=1000 / fps, blit=True)
        writer = FFMpegWriter(fps=fps)
        ani.save(save_path.replace(".png", ".mp4"), writer=writer, dpi=dpi)
        plt.close()
    else:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

# Example usage:
# stats = calculate_stats(ind_gate_times_sec, n_gates)
# plot_speed(ind_timesteps, ind_speed, gps[0], "Individual Speed Over Time", "./plots/speed_individual.png", animate=True)
# plot_angular_speed(ind_timesteps, ind_angular_speed, gps[0], "Individual Angular Speed Over Time", "./plots/angspeed_individual.png", animate=True)
# plot_actions(ind_timesteps, ass[0], gps[0], motor_colors, "./plots/actions_individual.png", animate=True)

# Example usage
    # 1) speed plot with gate lines and labels
    # 2) angspeed plot with gate lines and labels
    # 3) action plot
    # 6) speed plot animated, gate lines no, labels
    # 7) angspeed plot animated, gate lines no, labels
    # 8) action plot animated
def plot_speed_angspeed_actions(timesteps, speed, angular_speed, actions, gate_passes, save_dir,
                                motor_colors=['red', 'blue', 'green', 'orange', 'purple', 'brown'],
                                fps=100, width=864, height=700, dpi=200, gate_lines=False, gate_labels=False, 
                                gate_label_ylevel=11.0, fontsize=7, pad=0.05, offset_val=0.5, gate_line_alpha=0.5, 
                                alpha=1.0, color='blue', animate=False):
    """
    Plots speed, angular speed, and motor actions with gate lines and labels.

    Args:
        timesteps (np.ndarray): Array of timesteps.
        speed (np.ndarray): Array of speed values.
        angular_speed (np.ndarray): Array of angular speed values.
        actions (np.ndarray): Array of motor actions.
        gate_passes (np.ndarray): Array indicating gate passes.
        motor_colors (list): List of colors for motor actions.
        save_dir (str): Directory to save the plots.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1) Speed plot with gate lines and labels
    plot_speed(
        timesteps=timesteps,
        speed=speed,
        gate_passes=gate_passes,
        title="Speed Over Time with Gate Passages",
        save_path=os.path.join(save_dir, "speed_plot.png"),
        fps=fps, width=width, height=height, dpi=dpi, 
        gate_lines=gate_lines, gate_labels=gate_labels, 
        gate_label_ylevel=gate_label_ylevel, fontsize=fontsize, pad=pad, 
        offset_val=offset_val, gate_line_alpha=gate_line_alpha, alpha=alpha, 
        color=color, animate=animate
    )

    # 2) Angular speed plot with gate lines and labels
    plot_angular_speed(
        timesteps=timesteps,
        angular_speed=angular_speed,
        gate_passes=gate_passes,
        title="Angular Speed Over Time with Gate Passages",
        save_path=os.path.join(save_dir, "angular_speed_plot.png"),
        fps=fps, width=width, height=height, dpi=dpi, 
        gate_lines=gate_lines, gate_labels=gate_labels, 
        gate_label_ylevel=gate_label_ylevel, fontsize=fontsize, pad=pad, 
        offset_val=offset_val, gate_line_alpha=gate_line_alpha, alpha=alpha, 
        color=color, animate=animate
    )

    # 3) Action plot
    plot_actions(
        timesteps=timesteps,
        actions=actions,
        gate_passes=gate_passes,
        motor_colors=motor_colors,
        save_path=os.path.join(save_dir, "actions_plot.png"),
        fps=fps, width=width, height=height*2, dpi=dpi, animate=animate
    )

    # 4) speed plot compared to hexacopter
    # 5) angspeed compared to hexacopter
def plot_comparison(
    ind1_timesteps, ind1_data, ind1_gate_passes, 
    ind2_timesteps, ind2_data, ind2_gate_passes, 
    plot_function,
    save_dir, filename,
    gate_lines=False, gate_labels=False,
    title="Comparison Between Two Individuals", legend_labels=["Individual 1", "Individual 2"],
    ind1_color="orange", ind1_alpha=1.0, ind1_gate_line_alpha=0.5, ind1_gate_label_ylevel=12.0,
    ind2_color="green", ind2_alpha=1.0, ind2_gate_line_alpha=0.5, ind2_gate_label_ylevel=0.0,
    width=864, height=700, dpi=200
):
    """
    Plots a comparison between two individuals using the specified plot function.

    Args:
        ind1_timesteps: Timesteps for individual 1.
        ind1_data: Data for individual 1 (e.g., speed or angular speed).
        ind1_gate_passes: Gate passes for individual 1.
        ind2_timesteps: Timesteps for individual 2.
        ind2_data: Data for individual 2 (e.g., speed or angular speed).
        ind2_gate_passes: Gate passes for individual 2.
        plot_function: Function to use for plotting (e.g., plot_speed or plot_angular_speed).
        title: Title of the plot.
        save_dir: Directory to save the plot.
        filename: Filename for the saved plot.
        ind1_color: Color for individual 1 (default: "orange").
        ind1_alpha: Alpha for individual 1 (default: 1.0).
        ind1_gate_line_alpha: Gate line alpha for individual 1 (default: 0.5).
        ind1_gate_label_ylevel: Gate label y-level for individual 1 (default: 12.0).
        ind2_color: Color for individual 2 (default: "green").
        ind2_alpha: Alpha for individual 2 (default: 1.0).
        ind2_gate_line_alpha: Gate line alpha for individual 2 (default: 0.5).
        ind2_gate_label_ylevel: Gate label y-level for individual 2 (default: 0.0).
        width: Width of the plot (default: 864).
        height: Height of the plot (default: 700).
        dpi: DPI for the plot (default: 200).
    """
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi))
    plot_function(
        ind1_timesteps, ind1_data, ind1_gate_passes,
        title, save_path=None, ax=ax, color=ind1_color, alpha=ind1_alpha, gate_lines=gate_lines, gate_labels=gate_labels,
        gate_line_alpha=ind1_gate_line_alpha, gate_label_ylevel=ind1_gate_label_ylevel
    )
    plot_function(
        ind2_timesteps, ind2_data, ind2_gate_passes,
        title, save_path=None, ax=ax, color=ind2_color, alpha=ind2_alpha, gate_lines=gate_lines, gate_labels=gate_labels,
        gate_line_alpha=ind2_gate_line_alpha, gate_label_ylevel=ind2_gate_label_ylevel
    )
    ax.legend(legend_labels)
    fig.savefig(os.path.join(save_dir, filename), dpi=dpi, bbox_inches='tight')
    print(f"Plot saved to {os.path.join(save_dir, filename)}")



if __name__ == "__main__":

    # Define directories and configurations
    save_dir = "./plots/example_comparison/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load data for two individuals
    individual1_dir =  "/home/jed/workspaces/airevolve/data_backup/asym_slalom/asym_slalom4evo_logs_20250320_095329/gen40/ind964/"
    individual2_dir = "/home/jed/workspaces/airevolve/logs/hex_slalom/rep1/"
    ind1_policy_file = individual1_dir + "/policy.zip"
    ind2_policy_file = individual2_dir + "/policy.zip"
    individual1_body = individual1_dir + "/individual.npy"
    individual2_body = individual2_dir + "/individual.npy"
    individual1 = np.load(individual1_body)
    # individual2 = np.load(individual2_body)
    individual2 = np.array([[0.24, np.pi/3, 0.0, 0.0, 0.0, 1.0],
                        [0.24, 2*np.pi/3, 0.0, 0.0, 0.0, 0.0],
                        [0.24, np.pi, 0.0, 0.0, 0.0, 1.0],
                        [0.24, 4*np.pi/3, 0.0, 0.0, 0.0, 0.0],
                        [0.24, 5*np.pi/3, 0.0, 0.0, 0.0, 1.0],
                        [0.24, 0.0, 0.0, 0.0, 0.0, 0.0]])
                        
    gate_cfg = "slalom"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Extract simulation data for both individuals
    ind1_data = extract_simulation_data(individual1, ind1_policy_file, gate_cfg, device)
    ind2_data = extract_simulation_data(individual2, ind2_policy_file, gate_cfg, device)

    # Access extracted data
    ind1_speed = np.linalg.norm(ind1_data["velocities"], axis=1)
    ind1_angular_speed = np.linalg.norm(ind1_data["angular_velocities"], axis=1)
    ind1_timesteps = np.arange(len(ind1_speed))
    ind1_gate_passes = ind1_data["gate_passes"]
    ind1_actions = ind1_data["actions"]

    ind2_speed = np.linalg.norm(ind2_data["velocities"], axis=1)
    ind2_angular_speed = np.linalg.norm(ind2_data["angular_velocities"], axis=1)
    ind2_timesteps = np.arange(len(ind2_speed))
    ind2_gate_passes = ind2_data["gate_passes"]

    fps, width, height, dpi = 100, 864, 700, 200
    gate_label_ylevel = 11.0
    fontsize = 7
    pad = 0.05
    offset_val = 0.5
    gate_line_alpha = 0.5
    alpha = 1.0
    motor_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    color = 'blue'
    # Plot speed, angular speed, and actions for individual 1
    plot_speed_angspeed_actions(ind1_timesteps, ind1_speed, ind1_angular_speed, ind1_actions, ind1_gate_passes, save_dir=save_dir,
                                fps=fps, width=width, height=height, dpi=dpi, gate_lines=True, gate_labels=True, 
                                gate_label_ylevel=gate_label_ylevel, fontsize=fontsize, pad=pad, offset_val=offset_val, gate_line_alpha=gate_line_alpha, 
                                alpha=alpha, color=color, animate=False)
    # Plot speed comparison
    plot_comparison(
        ind1_timesteps, ind1_speed, ind1_gate_passes,
        ind2_timesteps, ind2_speed, ind2_gate_passes,
        plot_speed, save_dir, "speed_comparison.png",
        "Speed Comparison Between Two Individuals"  
    )

    # Example usage for angular speed comparison
    plot_comparison(
        ind1_timesteps, ind1_angular_speed, ind1_gate_passes,
        ind2_timesteps, ind2_angular_speed, ind2_gate_passes,
        plot_angular_speed, save_dir, "angular_speed_comparison.png",
        "Angular Speed Comparison Between Two Individuals"
    )

    plot_speed_angspeed_actions(ind1_timesteps, ind1_speed, ind1_angular_speed, ind1_actions, ind1_gate_passes, save_dir=save_dir,
                                fps=fps, width=width, height=height, dpi=dpi, gate_lines=True, gate_labels=False, 
                                gate_label_ylevel=gate_label_ylevel, fontsize=fontsize, pad=pad, offset_val=offset_val, gate_line_alpha=gate_line_alpha, 
                                alpha=alpha, color=color, animate=True)