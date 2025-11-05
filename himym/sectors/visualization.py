"""Visualization utilities for the evolutionary algorithm."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path


def visualize_grid_state(grid, population, current_positions, fitness_history, 
                        generation, config):
    """Visualize current population distribution across grid."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw grid
    for i in range(grid.grid_size + 1):
        x = i * grid.sector_width
        ax.axvline(x, color='black', linewidth=2)
        y = i * grid.sector_height
        ax.axhline(y, color='black', linewidth=2)
    
    # Color each sector by population count
    max_count = max(len(grid.sectors[i]) for i in range(9))
    
    for sector_id in range(9):
        count = len(grid.sectors[sector_id])
        x_min, x_max, y_min, y_max = grid.get_sector_bounds(sector_id)
        
        # Color intensity based on population
        alpha = 0.1 + 0.6 * (count / max_count if max_count > 0 else 0)
        color = plt.cm.Blues(alpha)
        
        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            facecolor=color,
            edgecolor='none'
        )
        ax.add_patch(rect)
        
        # Add sector ID and count
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        ax.text(center_x, center_y, f'S{sector_id}\n({count})',
               ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Plot individual positions
    for i, individual in enumerate(population):
        pos = current_positions[i]
        
        # Color by fitness
        fitness_norm = individual.fitness / max(fitness_history[-1], 0.001)
        color = plt.cm.Reds(min(fitness_norm, 1.0))
        
        ax.plot(pos[0], pos[1], 'o', color=color, markersize=8,
               markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlim(-0.1, config.world_size[0] + 0.1)
    ax.set_ylim(-0.1, config.world_size[1] + 0.1)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    ax.set_title(f'Generation {generation + 1} - Grid Population Distribution\n'
                f'Blue shading: population density | Red dots: individuals (intensity = fitness)',
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Create figures folder if it doesn't exist
    figures_path = Path(config.figures_folder)
    figures_path.mkdir(parents=True, exist_ok=True)
    
    save_path = f"{config.figures_folder}/grid_state_gen_{generation + 1:03d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_fitness_evolution(fitness_history, best_individual_history, config):
    """Plot fitness and p_local evolution."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Fitness evolution
    ax1.plot(range(1, len(fitness_history) + 1), 
            fitness_history, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best Fitness (Distance)")
    ax1.set_title("Fitness Evolution")
    ax1.grid(True, alpha=0.3)
    
    # p_local evolution (average per generation)
    p_local_history = []
    for gen_best in best_individual_history:
        p_local_history.append(gen_best.p_local)
    
    ax2.plot(range(1, len(p_local_history) + 1),
            p_local_history, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Best Individual's p_local")
    ax2.set_title("Mating Preference Evolution (0=distant, 1=local)")
    ax2.set_ylim(-0.1, 1.1)
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Create figures folder if it doesn't exist
    figures_path = Path(config.figures_folder)
    figures_path.mkdir(parents=True, exist_ok=True)
    
    save_path = f"{config.figures_folder}/grid_ea_evolution.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Evolution plots saved to {save_path}")


def plot_movement_analysis(movement_history, config):
    """Analyze and visualize movement patterns between sectors."""
    if len(movement_history) == 0:
        print("No movement data to visualize")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Movement frequency matrix (from_sector -> to_sector)
    movement_matrix = np.zeros((9, 9))
    for move in movement_history:
        from_s = move['from_sector']
        to_s = move['to_sector']
        movement_matrix[from_s, to_s] += 1
    
    # Plot heatmap
    im = ax1.imshow(movement_matrix, cmap='YlOrRd', aspect='auto')
    ax1.set_xlabel('To Sector', fontsize=12)
    ax1.set_ylabel('From Sector', fontsize=12)
    ax1.set_title('Sector-to-Sector Movement Frequency', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(9))
    ax1.set_yticks(range(9))
    
    # Add text annotations
    for i in range(9):
        for j in range(9):
            if movement_matrix[i, j] > 0:
                ax1.text(j, i, f'{int(movement_matrix[i, j])}',
                       ha='center', va='center', 
                       color='white' if movement_matrix[i, j] > movement_matrix.max()/2 else 'black')
    
    plt.colorbar(im, ax=ax1, label='Number of Movements')
    
    # Movement frequency per generation
    movements_per_gen = {}
    for move in movement_history:
        gen = move['generation']
        movements_per_gen[gen] = movements_per_gen.get(gen, 0) + 1
    
    gens = sorted(movements_per_gen.keys())
    counts = [movements_per_gen[g] for g in gens]
    
    ax2.bar([g+1 for g in gens], counts, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Number of Sector Changes', fontsize=12)
    ax2.set_title('Movement Activity Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Create figures folder if it doesn't exist
    figures_path = Path(config.figures_folder)
    figures_path.mkdir(parents=True, exist_ok=True)
    
    save_path = f"{config.figures_folder}/movement_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Movement analysis saved to {save_path}")
