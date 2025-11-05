#!/usr/bin/env python3
"""
Morphological fitness analysis and visualization.

This script:
1. Loads target robot JSONs and computes their 6D morphological descriptors
2. Generates random robots using tree genome
3. Computes fitness as distance to target descriptors
4. Visualizes fitness landscapes using PCA and various plots
"""
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Tuple, Callable, Any
import json
from pathlib import Path
import imageio.v2 as imageio

# Import ARIEL modules
from ariel.utils.graph_ops import load_robot_json_file
from ariel.utils.morphological_descriptor import MorphologicalMeasures
from ariel.ec.genotypes.tree.tree_genome import TreeGenome, TreeNode

import ariel.body_phenotypes.robogen_lite.config as config
from ariel.ec.a001 import Individual
import networkx as nx
import matplotlib.animation as animation

type Population = List[Individual]

# Set random seed for reproducibility
np.random.seed(42)

def compute_6d_descriptor(robot_graph: nx.DiGraph) -> np.ndarray:
        """Compute 6D morphological descriptor vector."""

        measures = MorphologicalMeasures(robot_graph)
        # Handle potential division by zero or missing P for non-2D robots
        try:
            P = measures.P if measures.is_2d else 0.0
        except:
            P = 0.0

        descriptor = np.array([
            measures.B,  # Branching
            measures.L,  # Limbs
            measures.E,  # Extensiveness (length of limbs)
            measures.S,  # Symmetry
            P,           # Proportion (2D only)
            measures.J   # Joints
        ])
        return descriptor


def load_target_robot(json_path: str):
    """Load target robots and compute their descriptors."""
    try:
        robot_graph = load_robot_json_file(json_path)
        descriptor = compute_6d_descriptor(robot_graph)

        # Extract name from path
        name = Path(json_path).stem
        print(f"Loaded {name}: {descriptor}")
        return descriptor

    except Exception as e:
        print(f"Error loading {json_path}: {e}")


def compute_fitness_scores(individual_descriptors, target_descriptors):
    """Compute fitness scores as mean of distances in each dimension to each target."""

    fitness_scores = []

    # Compute absolute differences in each dimension
    dimension_distances = np.abs(individual_descriptors - target_descriptors)
    # Take mean across dimensions to get fitness score
    mean_distances = np.mean(dimension_distances)
    # Convert to fitness (higher is better, so use negative distance)
    fitness = -mean_distances
    fitness_scores.append(fitness)

    return np.array(fitness_scores)


class MorphologyAnalyzer:
    """Analyze and visualize morphological fitness landscapes."""

    def __init__(self):
        self.target_descriptors = []
        self.target_names = []
        self.descriptors = []
        self.fitness_scores = []

    def compute_6d_descriptor(self, robot_graph) -> np.ndarray:
        """Compute 6D morphological descriptor vector."""

        measures = MorphologicalMeasures(robot_graph)
        # Handle potential division by zero or missing P for non-2D robots
        try:
            P = measures.P if measures.is_2d else 0.0
        except:
            P = 0.0

        descriptor = np.array([
            measures.B,  # Branching
            measures.L,  # Limbs
            measures.E,  # Extensiveness (length of limbs)
            measures.S,  # Symmetry
            P,           # Proportion (2D only)
            measures.J   # Joints
        ])
        return descriptor


    def load_target_robots(self, *json_paths: str):
        """Load target robots and compute their descriptors."""
        self.target_descriptors = []
        self.target_names = []

        for json_path in json_paths:
            try:
                robot_graph = load_robot_json_file(json_path)
                descriptor = self.compute_6d_descriptor(robot_graph)
                self.target_descriptors.append(descriptor)

                # Extract name from path
                name = Path(json_path).stem
                self.target_names.append(name)

                print(f"Loaded {name}: {descriptor}")

            except Exception as e:
                print(f"Error loading {json_path}: {e}")

        self.target_descriptors = np.array(self.target_descriptors)

    def generate_random_robot(self, max_depth: int = 3, branch_prob: float = 0.6) -> TreeGenome:
        """Generate a random robot using tree genome."""
        # Create root with CORE
        root = TreeNode(
            module_type=config.ModuleType.CORE,
            module_rotation=config.ModuleRotationsIdx.DEG_0
        )

        # Add random children
        self._add_random_children(root, max_depth, branch_prob)

        genome = TreeGenome(root)
        return genome

    def _add_random_children(self, node: TreeNode, max_depth: int, branch_prob: float):
        """Recursively add random children to a node."""
        if max_depth <= 0:
            return

        available_faces = node.available_faces()

        for face in available_faces:
            if np.random.random() < branch_prob:
                # Choose random module type (excluding CORE and NONE)
                module_types = [mt for mt in config.ModuleType
                              if mt not in {config.ModuleType.CORE, config.ModuleType.NONE}]
                module_type = np.random.choice(module_types)

                # Choose random rotation
                rotation = np.random.choice(list(config.ModuleRotationsIdx))

                # Create child node
                child = TreeNode(module_type=module_type, module_rotation=rotation)
                node._set_face(face, child)

                # Recursively add children with reduced depth
                self._add_random_children(child, max_depth - 1, branch_prob * 0.7)

    def load_population(self, population: Population, decoder: Callable[[Individual], nx.DiGraph]):
        """Load a population of robots and compute their descriptors."""
        self.descriptors = []
        descriptors = []
        for ind in population:
            robot_graph: nx.DiGraph = decoder(ind.genotype)
            descriptor = self.compute_6d_descriptor(robot_graph)
            descriptors.append(descriptor)

        self.descriptors = np.array(descriptors)

    def generate_random_population(self, n_robots: int = 100) -> List[np.ndarray]:
        """Generate a population of random robots and compute their descriptors."""
        self.descriptors = []
        print(f"Generating {n_robots} random robots...")
        descriptors = []

        for i in range(n_robots):
            if i % 20 == 0:
                print(f"Generated {i}/{n_robots} robots")

            # Generate random robot
            genome = self.generate_random_robot()

            # Decode to graph
            robot_graph = genome.to_digraph(genome) #! THIS DOES NOT WORK, to_digraph is a static method that needs a genome as an argument

            # Compute descriptor
            descriptor = self.compute_6d_descriptor(robot_graph)
            descriptors.append(descriptor)



        self.descriptors = np.array(descriptors)
        return self.descriptors

    def compute_fitness_scores(self):
        """Compute fitness scores as mean of distances in each dimension to each target."""
        if len(self.target_descriptors) == 0 or len(self.descriptors) == 0:
            raise ValueError("Need both target and random descriptors")

        self.fitness_scores = []

        for target_desc in self.target_descriptors:
            # Compute absolute differences in each dimension
            dimension_distances = np.abs(self.descriptors - target_desc)
            # Take mean across dimensions to get fitness score
            mean_distances = np.mean(dimension_distances, axis=1)
            # Convert to fitness (higher is better, so use negative distance)
            fitness = -mean_distances
            self.fitness_scores.append(fitness)

        self.fitness_scores = np.array(self.fitness_scores)

    def plot_target_descriptors_pca(self, return_fig: bool = False):
        """Plot target robots in PCA-reduced space."""
        if len(self.target_descriptors) == 0:
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Combine all descriptors for PCA fitting
        all_desc = np.vstack([self.target_descriptors, self.descriptors])
        pca = PCA(n_components=2)
        all_pca = pca.fit_transform(all_desc)

        target_pca = all_pca[:len(self.target_descriptors)]
        random_pca = all_pca[len(self.target_descriptors):]

        # Plot 1: PCA visualization
        axes[0].scatter(random_pca[:, 0], random_pca[:, 1],
                       alpha=0.3, c='purple', s=20, label='Evolved robots')

        colors = ['red', 'blue', 'green', 'orange']
        for i, (target, name) in enumerate(zip(target_pca, self.target_names)):
            color = colors[i % len(colors)]
            axes[0].scatter(target[0], target[1],
                           c=color, s=100, marker='*',
                           label=f'Target: {name}', edgecolors='black')

        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        axes[0].set_title('Morphological Space (PCA)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Feature importance
        feature_names = ['Branching', 'Limbs', 'Extensiveness', 'Symmetry', 'Proportion', 'Joints']

        pc1_importance = np.abs(pca.components_[0])
        pc2_importance = np.abs(pca.components_[1])

        x = np.arange(len(feature_names))
        width = 0.35

        axes[1].bar(x - width/2, pc1_importance, width, label='PC1', alpha=0.8)
        axes[1].bar(x + width/2, pc2_importance, width, label='PC2', alpha=0.8)

        axes[1].set_xlabel('Morphological Features')
        axes[1].set_ylabel('Absolute Component Weight')
        axes[1].set_title('PCA Feature Importance')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(feature_names, rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()
            return fig

    def plot_fitness_landscapes(self, return_fig: bool = False):
        """Plot fitness landscapes for each target."""
        if len(self.fitness_scores) == 0:
            self.compute_fitness_scores()

        # Use PCA for dimensionality reduction
        all_desc = np.vstack([self.target_descriptors, self.descriptors])
        pca = PCA(n_components=2)
        all_pca = pca.fit_transform(all_desc)

        target_pca = all_pca[:len(self.target_descriptors)]
        evolved_pca = all_pca[len(self.target_descriptors):]

        n_targets = len(self.target_names)
        fig, axes = plt.subplots(2, (n_targets + 1) // 2, figsize=(15, 10))
        if n_targets == 1:
            axes = [axes]
        axes = axes.flatten() if n_targets > 1 else axes[0]

        for i, (target_name, fitness) in enumerate(zip(self.target_names, self.fitness_scores)):
            ax = axes[i]

            # Create scatter plot with fitness as color
            scatter = ax.scatter(evolved_pca[:, 0], evolved_pca[:, 1],
                               c=fitness, cmap='viridis', alpha=0.6, s=30)

            # Mark target location
            ax.scatter(target_pca[i, 0], target_pca[i, 1],
                      c='red', s=200, marker='*',
                      edgecolors='black', linewidths=2,
                      label=f'Target: {target_name}')

            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            ax.set_title(f'Fitness Landscape: {target_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Fitness')

        # Hide unused subplots
        for j in range(n_targets, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()
            return fig

    def plot_fitness_distributions(self, return_fig: bool = False):
        """Plot fitness distributions for each target."""
        if len(self.fitness_scores) == 0:
            self.compute_fitness_scores()

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Compute target scores (perfect match = 0 distance = 0 fitness)
        target_scores = [0.0] * len(self.target_names)  # Perfect match has 0 mean distance

        # Plot 1: Fitness distributions
        for i, (target_name, fitness) in enumerate(zip(self.target_names, self.fitness_scores)):
            # Get target score info for legend
            target_score = target_scores[i]
            best_fitness = np.max(fitness)
            mean_fitness = np.mean(fitness)

            label = f'{target_name} (target: {target_score:.3f}, best: {best_fitness:.3f}, mean: {mean_fitness:.3f})'
            axes[0].hist(fitness, bins=30, alpha=0.7, label=label, density=True)

            # Mark target score with vertical line
            axes[0].axvline(target_score, color=f'C{i}', linestyle='--', linewidth=2, alpha=0.8)

        axes[0].set_xlabel('Fitness (negative mean distance)')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Fitness Distributions')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Best fitness per target
        best_fitness = [np.max(fitness) for fitness in self.fitness_scores]
        mean_fitness = [np.mean(fitness) for fitness in self.fitness_scores]

        x = np.arange(len(self.target_names))
        width = 0.35

        axes[1].bar(x - width/2, best_fitness, width, label='Best', alpha=0.8)
        axes[1].bar(x + width/2, mean_fitness, width, label='Mean', alpha=0.8)

        axes[1].set_xlabel('Target Robot')
        axes[1].set_ylabel('Fitness')
        axes[1].set_title('Fitness Statistics')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(self.target_names)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()
            return fig

    def analyze_morphological_diversity(self, return_fig: bool = False):
        """Analyze diversity in the random population."""
        if len(self.descriptors) == 0:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        feature_names = ['Branching', 'Limbs', 'Extensiveness', 'Symmetry', 'Proportion', 'Joints']

        for i, feature_name in enumerate(feature_names):
            ax = axes[i]

            # Plot distribution of random robots
            random_mean = np.mean(self.descriptors[:, i])
            random_std = np.std(self.descriptors[:, i])
            ax.hist(self.descriptors[:, i], bins=30, alpha=0.7,
                   density=True, label=f'Random robots (μ={random_mean:.3f}, σ={random_std:.3f})')

            # Mark target values
            for j, target_name in enumerate(self.target_names):
                target_value = self.target_descriptors[j, i]
                ax.axvline(target_value, color=f'C{j+1}', linestyle='--',
                          linewidth=2, label=f'{target_name}: {target_value:.3f}')

            ax.set_xlabel(feature_name)
            ax.set_ylabel('Density')
            ax.set_title(f'{feature_name} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()
            return fig

    def plot_pairwise_feature_landscapes(self, features: List[str] = None, return_fig: bool = False):
        """Plot fitness landscapes for pairwise combinations of selected morphological features.

        Parameters:
        -----------
        features : List[str], optional
            List of feature names to include. By default includes all features:
            ['Branching', 'Limbs', 'Extensiveness', 'Symmetry', 'Proportion', 'Joints']
        return_fig : bool, optional
            Whether to return the figure instead of showing it
        """
        if len(self.fitness_scores) == 0:
            self.compute_fitness_scores()

        # All available features with their indices
        all_feature_names = ['Branching', 'Limbs', 'Extensiveness', 'Symmetry', 'Proportion', 'Joints']

        # Use all features by default
        if features is None:
            features = all_feature_names.copy()

        # Get indices of selected features
        feature_indices = []
        selected_feature_names = []
        for feature in features:
            if feature in all_feature_names:
                idx = all_feature_names.index(feature)
                feature_indices.append(idx)
                selected_feature_names.append(feature)
            else:
                print(f"Warning: Feature '{feature}' not recognized. Available features: {all_feature_names}")

        n_features = len(feature_indices)
        if n_features < 2:
            print("Error: Need at least 2 features for pairwise plots")
            return None

        # Create all pairwise combinations
        pairs = []
        pair_names = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                pairs.append((feature_indices[i], feature_indices[j]))
                pair_names.append((selected_feature_names[i], selected_feature_names[j]))

        n_pairs = len(pairs)

        # Calculate optimal subplot grid
        if n_pairs <= 3:
            rows, cols = 1, n_pairs
        elif n_pairs <= 6:
            rows, cols = 2, 3
        elif n_pairs <= 12:
            rows, cols = 3, 4
        else:
            rows, cols = 3, 5  # For 15 pairs (all features)

        # Create subplot grid
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if n_pairs == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

        # Use the first target for fitness coloring (could be extended for multiple targets)
        fitness_values = self.fitness_scores[0] if len(self.fitness_scores) > 0 else np.zeros(len(self.descriptors))

        for idx, ((i, j), (name_i, name_j)) in enumerate(zip(pairs, pair_names)):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # Extract feature pairs
            x_values = self.descriptors[:, i]
            y_values = self.descriptors[:, j]

            # Create scatter plot with fitness as color
            scatter = ax.scatter(x_values, y_values, c=fitness_values,
                               cmap='viridis', alpha=0.6, s=30)
            # Add target locations for all targets
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for target_idx, (target_desc, target_name) in enumerate(zip(self.target_descriptors, self.target_names)):
                color = colors[target_idx % len(colors)]
                ax.scatter(target_desc[i], target_desc[j],
                          c=color, s=200, marker='*',
                          edgecolors='black', linewidths=2,
                          label=f'Target: {target_name}')

            ax.set_xlabel(name_i)
            ax.set_ylabel(name_j)
            ax.set_title(f'{name_i} vs {name_j}')
            ax.grid(True, alpha=0.3)

            # Add legend only to first subplot to avoid clutter
            if idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # Add colorbar to each subplot
            plt.colorbar(scatter, ax=ax, label='Fitness', shrink=0.8)

        # Hide unused subplots
        for idx in range(n_pairs, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()
            return fig

    def create_evolution_video(
        self,
        populations,
        decoder,
        plot_method_name: str,
        video_filename: str = "evolution_video.mp4",
        fps: int = 2,
        dpi: int = 100,
        **plot_kwargs
    ):
        """Create video by converting each plotted figure into an RGB array."""
        plot_method = getattr(self, plot_method_name)
        assert self.target_descriptors is not None, "Target descriptors not loaded"
        frames = []

        for generation_idx, current_pop in enumerate(populations):
            # Compute current data
            self.load_population(current_pop, decoder)
            self.compute_fitness_scores()

            # Generate figure (your existing plot method already returns fig)
            fig = plot_method(return_fig=True, **plot_kwargs)
            fig.suptitle(f"Generation {generation_idx}", fontsize=16)

            fig.canvas.draw()

            # --- Convert figure to numpy array ---
            buf = np.asarray(fig.canvas.buffer_rgba())
            frame = buf[..., :3].copy()  # drop alpha channel
            frames.append(frame)
            plt.close(fig)

        # --- Write video ---
        print("Saving {len(frames)} frames to {video_filename}...")
        imageio.mimsave(video_filename, frames, fps=fps, quality=8)
        print(" Saved: {video_filename}")


def main():
    """Main analysis function."""
    # Define target robot paths
    target_paths = [
        "examples/target_robots/small_robot_8.json",
        "examples/target_robots/medium_robot_15.json",
        "examples/target_robots/large_robot_25.json"
    ]

    analyzer = MorphologyAnalyzer()
    analyzer.load_target_robots(*target_paths)
    analyzer.generate_random_population(n_robots=200)
    analyzer.compute_fitness_scores()
    analyzer.plot_pairwise_feature_landscapes()
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()


