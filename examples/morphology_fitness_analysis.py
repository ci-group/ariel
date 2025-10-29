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
from typing import List, Tuple
import json
from pathlib import Path

# Import ARIEL modules
from ariel.utils.graph_ops import load_robot_json_file
from ariel.utils.morphological_descriptor import MorphologicalMeasures
from ariel.ec.genotypes.tree.tree_genome import TreeGenome, TreeNode
from ariel.body_phenotypes.robogen_lite.decoders.tree_decoder import to_digraph
import ariel.body_phenotypes.robogen_lite.config as config

# Set random seed for reproducibility
np.random.seed(42)

def compute_6d_descriptor(robot_graph) -> np.ndarray:
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
        self.random_descriptors = []
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


    def load_target_robots(self, json_paths: List[str]):
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

    def generate_random_population(self, n_robots: int = 100) -> List[np.ndarray]:
        """Generate a population of random robots and compute their descriptors."""
        print(f"Generating {n_robots} random robots...")
        descriptors = []

        for i in range(n_robots):
            if i % 20 == 0:
                print(f"Generated {i}/{n_robots} robots")

            try:
                # Generate random robot
                genome = self.generate_random_robot()

                # Decode to graph
                robot_graph = to_digraph(genome)

                # Compute descriptor
                descriptor = self.compute_6d_descriptor(robot_graph)
                descriptors.append(descriptor)

            except Exception as e:
                print(f"Error generating robot {i}: {e}")
                # Add zero descriptor for failed robots
                descriptors.append(np.zeros(6))

        self.random_descriptors = np.array(descriptors)
        return self.random_descriptors

    def compute_fitness_scores(self):
        """Compute fitness scores as mean of distances in each dimension to each target."""
        if len(self.target_descriptors) == 0 or len(self.random_descriptors) == 0:
            raise ValueError("Need both target and random descriptors")

        self.fitness_scores = []

        for target_desc in self.target_descriptors:
            # Compute absolute differences in each dimension
            dimension_distances = np.abs(self.random_descriptors - target_desc)
            # Take mean across dimensions to get fitness score
            mean_distances = np.mean(dimension_distances, axis=1)
            # Convert to fitness (higher is better, so use negative distance)
            fitness = -mean_distances
            self.fitness_scores.append(fitness)

        self.fitness_scores = np.array(self.fitness_scores)

    def plot_target_descriptors_pca(self):
        """Plot target robots in PCA-reduced space."""
        if len(self.target_descriptors) == 0:
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Combine all descriptors for PCA fitting
        all_desc = np.vstack([self.target_descriptors, self.random_descriptors])
        pca = PCA(n_components=2)
        all_pca = pca.fit_transform(all_desc)

        target_pca = all_pca[:len(self.target_descriptors)]
        random_pca = all_pca[len(self.target_descriptors):]

        # Plot 1: PCA visualization
        axes[0].scatter(random_pca[:, 0], random_pca[:, 1],
                       alpha=0.3, c='lightgray', s=20, label='Random robots')

        colors = ['red', 'blue', 'green', 'orange', 'purple']
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
        plt.show()

    def plot_fitness_landscapes(self):
        """Plot fitness landscapes for each target."""
        if len(self.fitness_scores) == 0:
            self.compute_fitness_scores()

        # Use PCA for dimensionality reduction
        all_desc = np.vstack([self.target_descriptors, self.random_descriptors])
        pca = PCA(n_components=2)
        all_pca = pca.fit_transform(all_desc)

        target_pca = all_pca[:len(self.target_descriptors)]
        random_pca = all_pca[len(self.target_descriptors):]

        n_targets = len(self.target_names)
        fig, axes = plt.subplots(2, (n_targets + 1) // 2, figsize=(15, 10))
        if n_targets == 1:
            axes = [axes]
        axes = axes.flatten() if n_targets > 1 else axes

        for i, (target_name, fitness) in enumerate(zip(self.target_names, self.fitness_scores)):
            ax = axes[i]

            # Create scatter plot with fitness as color
            scatter = ax.scatter(random_pca[:, 0], random_pca[:, 1],
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
        plt.show()

    def plot_fitness_distributions(self):
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
        plt.show()

    def analyze_morphological_diversity(self):
        """Analyze diversity in the random population."""
        if len(self.random_descriptors) == 0:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        feature_names = ['Branching', 'Limbs', 'Extensiveness', 'Symmetry', 'Proportion', 'Joints']

        for i, feature_name in enumerate(feature_names):
            ax = axes[i]

            # Plot distribution of random robots
            random_mean = np.mean(self.random_descriptors[:, i])
            random_std = np.std(self.random_descriptors[:, i])
            ax.hist(self.random_descriptors[:, i], bins=30, alpha=0.7,
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
        plt.show()


def main():
    """Main analysis function."""
    # Define target robot paths
    target_paths = [
        "examples/target_robots/small_robot_8.json",
        "examples/target_robots/medium_robot_15.json",
        "examples/target_robots/large_robot_25.json"
    ]

    analyzer = MorphologyAnalyzer()
    analyzer.load_target_robots(target_paths)
    analyzer.generate_random_population(n_robots=200)
    analyzer.compute_fitness_scores()
    analyzer.plot_target_descriptors_pca()
    analyzer.plot_fitness_landscapes()
    analyzer.plot_fitness_distributions()
    analyzer.analyze_morphological_diversity()
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
