"""
Test and visualize spawn collision detection

This script tests that robots don't overlap when spawned and creates
a visualization of their initial positions.
"""

import sys
sys.path.insert(0, '/home/ariel-himym/himym')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from himym.evolutionary_algo_discrete import SpatialEA
from ea_config import config
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko


def visualize_spawn_positions(spatial_ea, save_path=None):
    """
    Visualize robot spawn positions to verify no overlap.
    
    Args:
        spatial_ea: SpatialEA instance after spawn_population()
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get spawn positions
    positions = [ind.spawn_position for ind in spatial_ea.population]
    positions = np.array(positions)
    
    # Draw world boundaries
    world_rect = patches.Rectangle(
        (0, 0), 
        config.world_size[0], 
        config.world_size[1],
        linewidth=2, 
        edgecolor='black', 
        facecolor='lightgray',
        alpha=0.2
    )
    ax.add_patch(world_rect)
    
    # Draw spawn area boundaries
    spawn_width = config.spawn_x_max - config.spawn_x_min
    spawn_height = config.spawn_y_max - config.spawn_y_min
    spawn_rect = patches.Rectangle(
        (config.spawn_x_min, config.spawn_y_min),
        spawn_width,
        spawn_height,
        linewidth=2,
        edgecolor='green',
        facecolor='none',
        linestyle='--',
        label='Spawn area'
    )
    ax.add_patch(spawn_rect)
    
    # Draw robots as circles
    robot_radius = config.min_spawn_distance / 2  # Visual representation
    for i, pos in enumerate(positions):
        circle = patches.Circle(
            (pos[0], pos[1]),
            robot_radius,
            color='blue',
            alpha=0.6,
            edgecolor='darkblue',
            linewidth=2
        )
        ax.add_patch(circle)
        
        # Add robot number
        ax.text(pos[0], pos[1], str(i), 
               ha='center', va='center', 
               fontsize=8, fontweight='bold', color='white')
    
    # Calculate and display statistics
    min_dist = float('inf')
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(positions[i][:2] - positions[j][:2])
            min_dist = min(min_dist, dist)
    
    # Set plot properties
    ax.set_xlim(-0.2, config.world_size[0] + 0.2)
    ax.set_ylim(-0.2, config.world_size[1] + 0.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    
    title = f'Robot Spawn Positions (n={len(positions)})\n'
    title += f'Min Distance: {min_dist:.3f}m | Required: {config.min_spawn_distance:.3f}m'
    if min_dist >= config.min_spawn_distance:
        title += ' ✓'
        title_color = 'green'
    else:
        title += ' ✗'
        title_color = 'red'
    
    ax.set_title(title, fontsize=14, fontweight='bold', color=title_color)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def test_collision_detection(population_size=25):
    """
    Test collision detection and visualize results.
    
    Args:
        population_size: Number of robots to spawn
    """
    print(f"Testing spawn collision detection with {population_size} robots...")
    print(f"World size: {config.world_size}")
    print(f"Spawn area: X=[{config.spawn_x_min}, {config.spawn_x_max}], "
          f"Y=[{config.spawn_y_min}, {config.spawn_y_max}]")
    print(f"Minimum distance required: {config.min_spawn_distance}m\n")
    
    # Get robot specs
    temp_world = SimpleFlatWorld(config.world_size)
    temp_robot = gecko()
    temp_world.spawn(temp_robot.spec, spawn_position=[0, 0, 0])
    temp_model = temp_world.spec.compile()
    num_joints = temp_model.nu
    
    # Create spatial EA and spawn
    spatial_ea = SpatialEA(
        population_size=population_size,
        num_generations=1,
        num_joints=num_joints
    )
    spatial_ea.initialize_population()
    spatial_ea.spawn_population()
    
    # Verify no overlaps
    positions = [ind.spawn_position for ind in spatial_ea.population]
    min_dist = float('inf')
    overlaps = 0
    overlap_pairs = []
    
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(positions[i][:2] - positions[j][:2])
            min_dist = min(min_dist, dist)
            if dist < config.min_spawn_distance:
                overlaps += 1
                overlap_pairs.append((i, j, dist))
    
    # Print results
    print("="*60)
    print("COLLISION DETECTION TEST RESULTS")
    print("="*60)
    print(f"Robots spawned: {len(positions)}")
    print(f"Minimum distance found: {min_dist:.3f}m")
    print(f"Required minimum: {config.min_spawn_distance:.3f}m")
    print(f"Overlapping pairs: {overlaps}")
    
    if overlaps > 0:
        print("\nOverlapping robot pairs:")
        for i, j, dist in overlap_pairs:
            print(f"  Robots {i} and {j}: {dist:.3f}m (too close!)")
        print(f"\n✗ TEST FAILED - {overlaps} overlapping pairs detected!")
    else:
        print(f"\n✓ TEST PASSED - No overlapping robots!")
    
    print("="*60)
    
    # Visualize
    visualize_spawn_positions(
        spatial_ea,
        save_path=f"{config.figures_folder}/spawn_positions_test.png"
    )
    
    return spatial_ea, overlaps == 0


if __name__ == "__main__":
    # Test with different population sizes
    test_sizes = [10, 25, 50]
    
    for size in test_sizes:
        print(f"\n{'='*70}\n")
        spatial_ea, success = test_collision_detection(population_size=size)
        
        if not success:
            print(f"\n⚠️  Warning: Collision detection may need tuning for {size} robots")
            print("   Consider increasing world size or decreasing min_spawn_distance")
        
        input("\nPress Enter to test next population size...")
