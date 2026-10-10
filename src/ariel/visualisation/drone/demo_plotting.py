"""
Demo Plotting and Analysis Visualization

This module provides matplotlib-based plotting and analysis functions
for drone demonstration results including trajectory plots, performance
metrics, and comparative analysis.

Functions:
    plot_demo_results: Plot comprehensive results for a single demo
    plot_single_demo: Internal function for plotting individual demo
    compare_demos: Create comparison plots across multiple demos
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_demo_results(demos, demo_idx=None, save_plots=False):
    """
    Plot results from demonstrations.
    
    Args:
        demos: List of demo result dictionaries
        demo_idx: Index of demo to plot (None for all)
        save_plots: Whether to save plots to files
    """
    if not demos:
        print("No demo results to plot")
        return
    
    demos_to_plot = [demo_idx] if demo_idx is not None else range(len(demos))
    
    for idx in demos_to_plot:
        if idx >= len(demos):
            continue
            
        demo = demos[idx]
        plot_single_demo(demo, save_plots)

def plot_single_demo(demo, save_plots=False):
    """
    Plot results for a single demo.
    
    Args:
        demo: Demo result dictionary containing simulation data
        save_plots: Whether to save plot to file
    """
    states = demo['state_history']
    times = demo['time_history']
    controls = demo['control_history']
    
    if len(states) == 0:
        print(f"No data to plot for {demo['name']}")
        return
    
    # Extract data arrays
    positions = states[:, 0:3]
    velocities = states[:, 3:6]
    attitudes = states[:, 6:9] * 180/np.pi  # Convert to degrees
    angular_velocities = states[:, 9:12] * 180/np.pi  # Convert to deg/s
    
    # Create comprehensive figure with subplots
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(f"Demo: {demo['name']}", fontsize=16, fontweight='bold')
    
    # 1. 3D trajectory plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    # Plot target trajectory ghost line if available
    if 'target_trajectory' in demo:
        target_pos = demo['target_trajectory']
        ax1.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2], 
                'g--', linewidth=1, alpha=0.6, label='Target')
    
    # Plot actual trajectory
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Actual')
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, label='End')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_zlabel('Z [m]')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Position vs time
    ax2 = fig.add_subplot(2, 3, 2)
    
    # Plot target positions if available
    if 'target_trajectory' in demo:
        target_pos = demo['target_trajectory']
        target_times = demo['target_time_points']
        ax2.plot(target_times, target_pos[:, 0], 'r--', alpha=0.6, linewidth=1, label='Target X')
        ax2.plot(target_times, target_pos[:, 1], 'g--', alpha=0.6, linewidth=1, label='Target Y')
        ax2.plot(target_times, target_pos[:, 2], 'b--', alpha=0.6, linewidth=1, label='Target Z')
    
    # Plot actual positions
    ax2.plot(times, positions[:, 0], 'r-', label='Actual X')
    ax2.plot(times, positions[:, 1], 'g-', label='Actual Y') 
    ax2.plot(times, positions[:, 2], 'b-', label='Actual Z')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Position [m]')
    ax2.set_title('Position vs Time')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Velocity vs time
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(times, velocities[:, 0], 'r-', label='Vx')
    ax3.plot(times, velocities[:, 1], 'g-', label='Vy')
    ax3.plot(times, velocities[:, 2], 'b-', label='Vz')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Velocity [m/s]')
    ax3.set_title('Velocity vs Time')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Attitude vs time
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(times, attitudes[:, 0], 'r-', label='Roll')
    ax4.plot(times, attitudes[:, 1], 'g-', label='Pitch')
    ax4.plot(times, attitudes[:, 2], 'b-', label='Yaw')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Attitude [deg]')
    ax4.set_title('Attitude vs Time')
    ax4.legend()
    ax4.grid(True)
    
    # 5. Motor commands vs time
    ax5 = fig.add_subplot(2, 3, 5)
    num_motors = controls.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, num_motors))
    for i in range(num_motors):
        ax5.plot(times, controls[:, i], color=colors[i], label=f'Motor {i+1}')
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Motor Command [-1,1]')
    ax5.set_title(f'Motor Commands ({num_motors} motors)')
    ax5.legend()
    ax5.grid(True)
    
    # 6. Angular velocities vs time
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(times, angular_velocities[:, 0], 'r-', label='p (roll rate)')
    ax6.plot(times, angular_velocities[:, 1], 'g-', label='q (pitch rate)')
    ax6.plot(times, angular_velocities[:, 2], 'b-', label='r (yaw rate)')
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Angular Velocity [deg/s]')
    ax6.set_title('Angular Velocities vs Time')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    
    if save_plots:
        filename = f"demo_{demo['name'].replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    
    plt.show()

def compare_demos(demos, demo_indices=None):
    """
    Compare multiple demonstrations with overlay plots.
    
    Args:
        demos: List of demo result dictionaries
        demo_indices: List of demo indices to compare (None for all)
    """
    if demo_indices is None:
        demo_indices = range(len(demos))
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Demo Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(demo_indices)))
    
    for i, idx in enumerate(demo_indices):
        if idx >= len(demos):
            continue
            
        demo = demos[idx]
        states = demo['state_history']
        times = demo['time_history']
        
        if len(states) == 0:
            continue
        
        positions = states[:, 0:3]
        velocities = states[:, 3:6]
        attitudes = states[:, 6:9] * 180/np.pi
        
        color = colors[i]
        label = demo['name']
        
        # Position magnitude
        pos_mag = np.linalg.norm(positions, axis=1)
        axes[0,0].plot(times, pos_mag, color=color, label=label)
        
        # Velocity magnitude  
        vel_mag = np.linalg.norm(velocities, axis=1)
        axes[0,1].plot(times, vel_mag, color=color, label=label)
        
        # Altitude
        axes[1,0].plot(times, positions[:, 2], color=color, label=label)
        
        # Yaw angle
        axes[1,1].plot(times, attitudes[:, 2], color=color, label=label)
    
    # Configure subplot properties
    axes[0,0].set_title('Position Magnitude')
    axes[0,0].set_xlabel('Time [s]')
    axes[0,0].set_ylabel('|Position| [m]')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    axes[0,1].set_title('Velocity Magnitude')
    axes[0,1].set_xlabel('Time [s]')
    axes[0,1].set_ylabel('|Velocity| [m/s]')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    axes[1,0].set_title('Altitude')
    axes[1,0].set_xlabel('Time [s]')
    axes[1,0].set_ylabel('Z [m]')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    axes[1,1].set_title('Yaw Angle')
    axes[1,1].set_xlabel('Time [s]')
    axes[1,1].set_ylabel('Yaw [deg]')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_performance_metrics(demos):
    """
    Create a performance metrics summary plot.
    
    Args:
        demos: List of demo result dictionaries
    """
    if not demos:
        print("No demos to analyze")
        return
    
    # Extract metrics for each demo
    demo_names = []
    max_positions = []
    max_velocities = []
    max_attitudes = []
    final_positions = []
    simulation_times = []
    num_motors = []
    
    for demo in demos:
        states = demo['state_history']
        if len(states) == 0:
            continue
            
        demo_names.append(demo['name'])
        
        positions = states[:, 0:3]
        velocities = states[:, 3:6] 
        attitudes = states[:, 6:9] * 180/np.pi
        
        max_positions.append(np.max(np.abs(positions), axis=0))
        max_velocities.append(np.max(np.abs(velocities), axis=0))
        max_attitudes.append(np.max(np.abs(attitudes), axis=0))
        final_positions.append(positions[-1])
        simulation_times.append(demo['simulation_time'])
        num_motors.append(demo['num_motors'])
    
    if not demo_names:
        print("No valid demo data for metrics")
        return
    
    # Create metrics visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Performance Metrics Summary', fontsize=16, fontweight='bold')
    
    x_pos = np.arange(len(demo_names))
    
    # Max position components
    max_pos_array = np.array(max_positions)
    axes[0,0].bar(x_pos - 0.2, max_pos_array[:, 0], 0.2, label='X', alpha=0.8)
    axes[0,0].bar(x_pos, max_pos_array[:, 1], 0.2, label='Y', alpha=0.8)  
    axes[0,0].bar(x_pos + 0.2, max_pos_array[:, 2], 0.2, label='Z', alpha=0.8)
    axes[0,0].set_title('Max Position [m]')
    axes[0,0].set_xticks(x_pos)
    axes[0,0].set_xticklabels(demo_names, rotation=45)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Max velocity components
    max_vel_array = np.array(max_velocities)
    axes[0,1].bar(x_pos - 0.2, max_vel_array[:, 0], 0.2, label='Vx', alpha=0.8)
    axes[0,1].bar(x_pos, max_vel_array[:, 1], 0.2, label='Vy', alpha=0.8)
    axes[0,1].bar(x_pos + 0.2, max_vel_array[:, 2], 0.2, label='Vz', alpha=0.8)
    axes[0,1].set_title('Max Velocity [m/s]')
    axes[0,1].set_xticks(x_pos)
    axes[0,1].set_xticklabels(demo_names, rotation=45)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Max attitude components
    max_att_array = np.array(max_attitudes)
    axes[0,2].bar(x_pos - 0.2, max_att_array[:, 0], 0.2, label='Roll', alpha=0.8)
    axes[0,2].bar(x_pos, max_att_array[:, 1], 0.2, label='Pitch', alpha=0.8)
    axes[0,2].bar(x_pos + 0.2, max_att_array[:, 2], 0.2, label='Yaw', alpha=0.8)
    axes[0,2].set_title('Max Attitude [deg]')
    axes[0,2].set_xticks(x_pos)
    axes[0,2].set_xticklabels(demo_names, rotation=45)
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Final position error magnitude
    final_pos_array = np.array(final_positions)
    final_pos_mag = np.linalg.norm(final_pos_array, axis=1)
    axes[1,0].bar(x_pos, final_pos_mag, alpha=0.8, color='orange')
    axes[1,0].set_title('Final Position Error [m]')
    axes[1,0].set_xticks(x_pos)
    axes[1,0].set_xticklabels(demo_names, rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    # Simulation performance
    sim_times = np.array(simulation_times)
    axes[1,1].bar(x_pos, sim_times, alpha=0.8, color='green')
    axes[1,1].set_title('Simulation Time [s]')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels(demo_names, rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    
    # Number of motors
    motors = np.array(num_motors)
    axes[1,2].bar(x_pos, motors, alpha=0.8, color='purple')
    axes[1,2].set_title('Number of Motors')
    axes[1,2].set_xticks(x_pos)
    axes[1,2].set_xticklabels(demo_names, rotation=45)
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_trajectory_3d_comparison(demos, demo_indices=None):
    """
    Create a 3D comparison plot of multiple trajectories.
    
    Args:
        demos: List of demo result dictionaries
        demo_indices: List of demo indices to compare (None for all)
    """
    if demo_indices is None:
        demo_indices = range(len(demos))
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(demo_indices)))
    
    for i, idx in enumerate(demo_indices):
        if idx >= len(demos):
            continue
            
        demo = demos[idx]
        states = demo['state_history']
        
        if len(states) == 0:
            continue
        
        positions = states[:, 0:3]
        color = colors[i]
        label = demo['name']
        
        # Plot target trajectory ghost line if available
        if 'target_trajectory' in demo:
            target_pos = demo['target_trajectory']
            ax.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2], 
                    color=color, linewidth=1, linestyle='--', alpha=0.4, 
                    label=f'{label} (Target)')
        
        # Plot actual trajectory
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                color=color, linewidth=2, label=f'{label} (Actual)', alpha=0.8)
        
        # Mark start and end points
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                   c=color, s=100, marker='o', alpha=0.9)
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                   c=color, s=100, marker='s', alpha=0.9)
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('3D Trajectory Comparison\n(Circles: Start, Squares: End)')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()