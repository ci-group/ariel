#!/usr/bin/env python3
"""
Plotly-based Morphological Analysis

This module provides interactive morphological analysis and visualization
using Plotly for web dashboards. It mirrors the functionality of
MorphologyAnalyzer but returns Plotly figures instead of matplotlib plots.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Tuple, Callable, Any
from pathlib import Path
import networkx as nx

# Import ARIEL modules
from ariel.utils.graph_ops import load_robot_json_file
from ariel.utils.morphological_descriptor import MorphologicalMeasures
from ariel.ec.genotypes.tree.tree_genome import TreeGenome, TreeNode
import ariel.body_phenotypes.robogen_lite.config as config
from ariel.ec.a001 import Individual

type Population = List[Individual]


class PlotlyMorphologyAnalyzer:
    """Analyze and visualize morphological fitness landscapes using Plotly."""

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

    def load_population(self, population: Population, decoder: Callable[[Individual], nx.DiGraph]):
        """Load a population of robots and compute their descriptors."""
        self.descriptors = []
        descriptors = []
        for ind in population:
            robot_graph: nx.DiGraph = decoder(ind)
            descriptor = self.compute_6d_descriptor(robot_graph)
            descriptors.append(descriptor)

        self.descriptors = np.array(descriptors)

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

    def plot_target_descriptors_pca(self) -> go.Figure:
        """Plot target robots in PCA-reduced space using Plotly."""
        if len(self.target_descriptors) == 0:
            return go.Figure()

        # Combine all descriptors for PCA fitting
        all_desc = np.vstack([self.target_descriptors, self.descriptors])
        pca = PCA(n_components=2)
        all_pca = pca.fit_transform(all_desc)

        target_pca = all_pca[:len(self.target_descriptors)]
        random_pca = all_pca[len(self.target_descriptors):]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Morphological Space (PCA)', 'PCA Feature Importance']
        )

        # Plot 1: PCA visualization
        fig.add_trace(
            go.Scatter(
                x=random_pca[:, 0],
                y=random_pca[:, 1],
                mode='markers',
                name='Evolved robots',
                marker=dict(color='purple', size=8, opacity=0.6)
            ),
            row=1, col=1
        )

        colors = ['red', 'blue', 'green', 'orange']
        for i, (target, name) in enumerate(zip(target_pca, self.target_names)):
            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=[target[0]],
                    y=[target[1]],
                    mode='markers',
                    name=f'Target: {name}',
                    marker=dict(color=color, size=15, symbol='star', line=dict(width=2, color='black'))
                ),
                row=1, col=1
            )

        # Plot 2: Feature importance
        feature_names = ['Branching', 'Limbs', 'Extensiveness', 'Symmetry', 'Proportion', 'Joints']
        pc1_importance = np.abs(pca.components_[0])
        pc2_importance = np.abs(pca.components_[1])

        fig.add_trace(
            go.Bar(x=feature_names, y=pc1_importance, name='PC1', opacity=0.8),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=feature_names, y=pc2_importance, name='PC2', opacity=0.8),
            row=1, col=2
        )

        fig.update_xaxes(title_text=f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', row=1, col=1)
        fig.update_yaxes(title_text=f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', row=1, col=1)
        fig.update_xaxes(title_text='Morphological Features', row=1, col=2)
        fig.update_yaxes(title_text='Absolute Component Weight', row=1, col=2)

        fig.update_layout(height=600, title_text="PCA Analysis")

        return fig

    def plot_fitness_landscapes(self) -> go.Figure:
        """Plot fitness landscapes for each target using Plotly."""
        if len(self.fitness_scores) == 0:
            self.compute_fitness_scores()

        # Use PCA for dimensionality reduction
        all_desc = np.vstack([self.target_descriptors, self.descriptors])
        pca = PCA(n_components=2)
        all_pca = pca.fit_transform(all_desc)

        target_pca = all_pca[:len(self.target_descriptors)]
        evolved_pca = all_pca[len(self.target_descriptors):]

        n_targets = len(self.target_names)
        fig = make_subplots(
            rows=1, cols=n_targets,
            subplot_titles=[f'Fitness: {name}' for name in self.target_names]
        )

        colors = ['red', 'blue', 'green', 'orange']
        for i, (target_name, fitness) in enumerate(zip(self.target_names, self.fitness_scores)):
            col = i + 1

            # Create scatter plot with fitness as color
            fig.add_trace(
                go.Scatter(
                    x=evolved_pca[:, 0],
                    y=evolved_pca[:, 1],
                    mode='markers',
                    marker=dict(
                        color=fitness,
                        colorscale='Viridis',
                        size=8,
                        opacity=0.7,
                        colorbar=dict(title='Fitness') if i == 0 else None
                    ),
                    name=f'Population - {target_name}',
                    showlegend=False
                ),
                row=1, col=col
            )

            # Mark target location
            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=[target_pca[i, 0]],
                    y=[target_pca[i, 1]],
                    mode='markers',
                    name=f'Target: {target_name}',
                    marker=dict(color=color, size=15, symbol='star', line=dict(width=2, color='black')),
                    showlegend=(i == 0)
                ),
                row=1, col=col
            )

        fig.update_layout(height=500, title_text="Fitness Landscapes")

        return fig

    def plot_fitness_distributions(self) -> go.Figure:
        """Plot fitness distributions for each target using Plotly."""
        if len(self.fitness_scores) == 0:
            self.compute_fitness_scores()

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Fitness Distributions', 'Fitness Statistics']
        )

        # Compute target scores (perfect match = 0 distance = 0 fitness)
        target_scores = [0.0] * len(self.target_names)

        # Plot 1: Fitness distributions
        for i, (target_name, fitness) in enumerate(zip(self.target_names, self.fitness_scores)):
            target_score = target_scores[i]
            best_fitness = np.max(fitness)
            mean_fitness = np.mean(fitness)

            fig.add_trace(
                go.Histogram(
                    x=fitness,
                    name=f'{target_name} (target: {target_score:.3f}, best: {best_fitness:.3f}, mean: {mean_fitness:.3f})',
                    opacity=0.7,
                    nbinsx=30
                ),
                row=1, col=1
            )

            # Mark target score with vertical line
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            fig.add_vline(
                x=target_score,
                line=dict(color=colors[i % len(colors)], dash='dash', width=2),
                row=1, col=1
            )

        # Plot 2: Best fitness per target
        best_fitness = [np.max(fitness) for fitness in self.fitness_scores]
        mean_fitness = [np.mean(fitness) for fitness in self.fitness_scores]

        fig.add_trace(
            go.Bar(x=self.target_names, y=best_fitness, name='Best', opacity=0.8),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=self.target_names, y=mean_fitness, name='Mean', opacity=0.8),
            row=1, col=2
        )

        fig.update_xaxes(title_text='Fitness (negative mean distance)', row=1, col=1)
        fig.update_yaxes(title_text='Count', row=1, col=1)
        fig.update_xaxes(title_text='Target Robot', row=1, col=2)
        fig.update_yaxes(title_text='Fitness', row=1, col=2)

        fig.update_layout(height=500, title_text="Fitness Distributions")

        return fig

    def analyze_morphological_diversity(self) -> go.Figure:
        """Analyze diversity in the population using Plotly."""
        if len(self.descriptors) == 0:
            return go.Figure()

        feature_names = ['Branching', 'Limbs', 'Extensiveness', 'Symmetry', 'Proportion', 'Joints']

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=feature_names
        )

        for i, feature_name in enumerate(feature_names):
            row = (i // 3) + 1
            col = (i % 3) + 1

            # Plot distribution of evolved robots
            feature_data = self.descriptors[:, i]
            random_mean = np.mean(feature_data)
            random_std = np.std(feature_data)

            fig.add_trace(
                go.Histogram(
                    x=feature_data,
                    name=f'Evolved robots (μ={random_mean:.3f}, σ={random_std:.3f})',
                    opacity=0.7,
                    nbinsx=30,
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )

            # Mark target values
            colors = ['red', 'blue', 'green', 'orange']
            for j, target_name in enumerate(self.target_names):
                target_value = self.target_descriptors[j, i]
                fig.add_vline(
                    x=target_value,
                    line=dict(color=colors[j % len(colors)], dash='dash', width=2),
                    row=row, col=col
                )

        fig.update_layout(height=600, title_text="Morphological Diversity Analysis")

        return fig

    def plot_pairwise_feature_landscapes(self, features: List[str] = None) -> go.Figure:
        """Plot fitness landscapes for pairwise combinations of selected morphological features using Plotly.

        Parameters:
        -----------
        features : List[str], optional
            List of feature names to include. By default includes all features:
            ['Branching', 'Limbs', 'Extensiveness', 'Symmetry', 'Proportion', 'Joints']
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
            return go.Figure()

        # Create all pairwise combinations - limit to 6 key pairs for display
        key_pairs = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5)]
        key_pairs = [(feature_indices[i], feature_indices[j]) for i, j in key_pairs 
                     if i < len(feature_indices) and j < len(feature_indices)][:6]
        
        pair_names = [(all_feature_names[i], all_feature_names[j]) for i, j in key_pairs]

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f'{name_i} vs {name_j}' for name_i, name_j in pair_names]
        )

        # Use the first target for fitness coloring
        fitness_values = self.fitness_scores[0] if len(self.fitness_scores) > 0 else np.zeros(len(self.descriptors))

        for idx, ((i, j), (name_i, name_j)) in enumerate(zip(key_pairs, pair_names)):
            row = (idx // 3) + 1
            col = (idx % 3) + 1

            # Extract feature pairs
            x_values = self.descriptors[:, i]
            y_values = self.descriptors[:, j]

            # Create scatter plot with fitness as color
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='markers',
                    marker=dict(
                        color=fitness_values,
                        colorscale='Viridis',
                        size=6,
                        opacity=0.7,
                        colorbar=dict(title='Fitness') if idx == 0 else None
                    ),
                    name='Population',
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )

            # Add target locations for all targets
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for target_idx, (target_desc, target_name) in enumerate(zip(self.target_descriptors, self.target_names)):
                color = colors[target_idx % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=[target_desc[i]],
                        y=[target_desc[j]],
                        mode='markers',
                        name=f'Target: {target_name}' if idx == 0 else None,
                        marker=dict(color=color, size=12, symbol='star', line=dict(width=2, color='black')),
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col
                )

            fig.update_xaxes(title_text=name_i, row=row, col=col)
            fig.update_yaxes(title_text=name_j, row=row, col=col)

        fig.update_layout(height=600, title_text="Pairwise Feature Landscapes")

        return fig