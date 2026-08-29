"""
Main interface for evolutionary analysis and visualization.

This module provides high-level classes and functions that orchestrate
the various components for comprehensive evolutionary analysis.
"""

from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from pathlib import Path

from .population_filtering import (
    get_top_k_individuals,
    get_generation_fitness_statistics,
    flatten_population_across_generations,
    filter_by_unique_individuals
)
from .grid_generators import (
    generate_best_individuals_fitness_grid,
    calculate_optimal_grid_size
)
from .evolution_plotters import (
    plot_best_individuals_ever,
    plot_best_individuals_per_generation,
    plot_population,
    create_evolution_summary_plot,
    save_evolution_plots
)
from .animation_utils import (
    animate_grids,
    AnimationBuilder,
    get_recommended_settings
)


class EvolutionAnalyzer:
    """
    Main class for analyzing evolutionary runs and creating visualizations.
    
    This class provides a high-level interface for all evolutionary analysis
    tasks, from basic statistics to complex visualizations and animations.
    """
    
    def __init__(
        self,
        population_data: npt.NDArray,
        fitness_data: npt.NDArray,
        parameter_limits: Optional[npt.NDArray] = None,
        generation_labels: Optional[List[str]] = None
    ):
        """
        Initialize the evolution analyzer.
        
        Args:
            population_data: Array of shape (n_generations, pop_size, n_arms, n_params)
            fitness_data: Array of shape (n_generations, pop_size)
            parameter_limits: Optional parameter bounds for similarity calculations
            generation_labels: Optional labels for generations
        """
        self.population_data = np.asarray(population_data)
        self.fitness_data = np.asarray(fitness_data)
        self.parameter_limits = parameter_limits
        self.generation_labels = generation_labels
        
        # Validate input shapes
        if self.population_data.ndim != 4:
            raise ValueError("population_data must be 4D: (generations, pop_size, n_arms, n_params)")
        if self.fitness_data.ndim != 2:
            raise ValueError("fitness_data must be 2D: (generations, pop_size)")
        
        n_gens_pop, pop_size_pop = self.population_data.shape[:2]
        n_gens_fit, pop_size_fit = self.fitness_data.shape
        
        if n_gens_pop != n_gens_fit or pop_size_pop != pop_size_fit:
            raise ValueError("Population and fitness data shapes don't match")
        
        self.n_generations = n_gens_pop
        self.pop_size = pop_size_pop
        self.n_arms = self.population_data.shape[2]
        self.n_params = self.population_data.shape[3]
        
        # Cache for computed statistics
        self._fitness_stats = None
        self._best_individuals = None
        self._best_fitnesses = None
    
    @property
    def fitness_statistics(self) -> Dict[str, Any]:
        """Get fitness statistics across all generations."""
        if self._fitness_stats is None:
            self._fitness_stats = get_generation_fitness_statistics(self.fitness_data)
        return self._fitness_stats
    
    @property
    def best_individuals_per_generation(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """Get the best individual from each generation."""
        if self._best_individuals is None:
            from .population_filtering import select_best_per_generation
            self._best_individuals, self._best_fitnesses = select_best_per_generation(
                self.population_data, self.fitness_data
            )
        return self._best_individuals, self._best_fitnesses
    
    def get_top_individuals_overall(
        self,
        k: int = 10,
        similarity_threshold: Optional[float] = None
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Get the top k individuals across all generations.
        
        Args:
            k: Number of individuals to return
            similarity_threshold: If provided, ensure diversity
            
        Returns:
            Tuple of (top_individuals, top_fitnesses)
        """
        flat_pop, flat_fit = flatten_population_across_generations(
            self.population_data, self.fitness_data, remove_nan=True
        )
        
        top_individuals, top_fitnesses, _ = get_top_k_individuals(
            flat_pop, flat_fit, k, reverse=False,
            similarity_threshold=similarity_threshold,
            parameter_limits=self.parameter_limits
        )
        
        return np.array(top_individuals), np.array(top_fitnesses)
    
    def analyze_diversity(self, generation: Optional[int] = None) -> Dict[str, float]:
        """
        Analyze population diversity for a specific generation or overall.
        
        Args:
            generation: Specific generation to analyze (if None, analyzes all)
            
        Returns:
            Dictionary with diversity metrics
        """
        if self.parameter_limits is None:
            raise ValueError("parameter_limits required for diversity analysis")
        
        # Import edit distance functions
        from ariel.ec.drone.evaluators.edit_distance import compute_edit_distance
        
        if generation is not None:
            # Analyze specific generation
            population = self.population_data[generation]
            fitness = self.fitness_data[generation]
            
            # Remove NaN individuals
            valid_mask = ~np.isnan(fitness)
            population = population[valid_mask]
            fitness = fitness[valid_mask]
        else:
            # Analyze across all generations
            population, fitness = flatten_population_across_generations(
                self.population_data, self.fitness_data, remove_nan=True
            )
        
        if len(population) == 0:
            return {'mean_distance': 0.0, 'std_distance': 0.0, 'diversity_index': 0.0}
        
        # Extract min/max values from parameter_limits
        # Assuming parameter_limits is (min_vals, max_vals) tuple
        if isinstance(self.parameter_limits, tuple) and len(self.parameter_limits) == 2:
            min_vals, max_vals = self.parameter_limits
        else:
            # Fallback: use data bounds
            min_vals = np.nanmin(population, axis=(0, 1))
            max_vals = np.nanmax(population, axis=(0, 1))
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = compute_edit_distance(population[i], population[j], min_vals, max_vals)
                distances.append(dist)
        
        distances = np.array(distances)
        
        return {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'diversity_index': np.mean(distances) / self.n_arms,  # Normalized
            'num_individuals': len(population)
        }
    
    def create_summary_report(self) -> Dict[str, Any]:
        """
        Create a comprehensive summary report of the evolutionary run.
        
        Returns:
            Dictionary with summary statistics and analysis
        """
        # Basic statistics
        fitness_stats = self.fitness_statistics
        best_inds, best_fits = self.best_individuals_per_generation
        
        # Overall best
        best_overall_idx = np.argmax(best_fits)
        best_overall_fitness = best_fits[best_overall_idx]
        best_overall_generation = best_overall_idx
        
        # Convergence analysis
        final_gen_mean = fitness_stats['mean'][-1]
        initial_gen_mean = fitness_stats['mean'][0]
        improvement = final_gen_mean - initial_gen_mean
        
        # Diversity analysis
        try:
            initial_diversity = self.analyze_diversity(0)
            final_diversity = self.analyze_diversity(-1)
            overall_diversity = self.analyze_diversity()
        except (ValueError, ImportError):
            initial_diversity = final_diversity = overall_diversity = {}
        
        report = {
            'run_statistics': {
                'n_generations': self.n_generations,
                'population_size': self.pop_size,
                'n_arms': self.n_arms,
                'n_parameters': self.n_params
            },
            'fitness_summary': {
                'best_overall': float(best_overall_fitness),
                'best_generation': int(best_overall_generation),
                'final_mean': float(final_gen_mean),
                'initial_mean': float(initial_gen_mean),
                'improvement': float(improvement),
                'final_max': float(fitness_stats['max'][-1]),
                'final_min': float(fitness_stats['min'][-1]),
                'final_std': float(fitness_stats['std'][-1])
            },
            'diversity_analysis': {
                'initial': initial_diversity,
                'final': final_diversity,
                'overall': overall_diversity
            },
            'convergence_metrics': {
                'fitness_range_final': float(fitness_stats['max'][-1] - fitness_stats['min'][-1]),
                'fitness_range_initial': float(fitness_stats['max'][0] - fitness_stats['min'][0]),
                'generations_to_best': int(best_overall_generation),
                'improvement_rate': float(improvement / self.n_generations)
            }
        }
        
        return report
    
    def plot_fitness_evolution(
        self,
        figsize: Tuple[float, float] = (12, 8),
        show_stats: bool = True
    ) -> plt.Figure:
        """
        Create a comprehensive fitness evolution plot.
        
        Args:
            figsize: Figure size
            show_stats: Whether to show statistical bands
            
        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        stats = self.fitness_statistics
        generations = np.arange(self.n_generations)
        
        # Plot mean fitness
        ax.plot(generations, stats['mean'], label='Mean', linewidth=2, color='blue')
        
        if show_stats:
            # Add standard deviation bands
            ax.fill_between(
                generations,
                stats['mean'] - stats['std'],
                stats['mean'] + stats['std'],
                alpha=0.3, color='blue', label='±1 Std'
            )
        
        # Plot max and min
        ax.plot(generations, stats['max'], label='Best', linestyle='--', color='green')
        ax.plot(generations, stats['min'], label='Worst', linestyle='--', color='red')
        
        # Highlight best overall
        best_inds, best_fits = self.best_individuals_per_generation
        best_gen = np.argmax(best_fits)
        ax.scatter([best_gen], [best_fits[best_gen]], 
                  color='gold', s=100, zorder=5, label='Best Overall')
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('Fitness Evolution Over Generations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_animation(
        self,
        animation_type: str = 'best_with_fitness',
        output_path: Optional[str] = None,
        **kwargs
    ) -> 'animation.ArtistAnimation':
        """
        Create an animation of the evolutionary process.
        
        Args:
            animation_type: Type of animation ('best_with_fitness', 'top_individuals', 'population')
            output_path: Path to save animation
            **kwargs: Additional arguments for animation creation
            
        Returns:
            matplotlib ArtistAnimation object
        """
        if animation_type == 'best_with_fitness':
            grids = generate_best_individuals_fitness_grid(
                list(self.population_data), 
                list(self.fitness_data),
                **kwargs
            )
        elif animation_type == 'top_individuals':
            # Create grids showing top individuals from each generation
            grids = []
            for gen in range(self.n_generations):
                pop = self.population_data[gen]
                fit = self.fitness_data[gen]
                
                # Get top individuals
                top_inds, top_fits, _ = get_top_k_individuals(pop, fit, 9)
                
                # Create grid (this would need to be implemented)
                grid = self._create_simple_grid(top_inds, top_fits, (3, 3), **kwargs)
                grids.append(grid)
        else:
            raise ValueError(f"Unknown animation type: {animation_type}")
        
        # Get recommended settings
        settings = get_recommended_settings(len(grids))
        settings.update(kwargs)
        
        return animate_grids(grids, output_path=output_path, **settings)
    
    def _create_simple_grid(self, individuals, fitnesses, grid_size, twod=True, **kwargs):
        """Helper method to create a simple visualization grid."""
        import functools
        from ariel.ec.drone.inspection.drone_visualizer import DroneVisualizer
        
        visualizer = DroneVisualizer()
        rows, cols = grid_size
        grid = []
        
        for i, (ind, fit) in enumerate(zip(individuals, fitnesses)):
            if i >= rows * cols:
                break
                
            if twod:
                viz_func = functools.partial(
                    visualizer.plot_2d,
                    genome_data=ind,
                    title=f"Fitness: {np.round(fit, 2)}",
                    **kwargs
                )
            else:
                viz_func = functools.partial(
                    visualizer.plot_3d,
                    genome_data=ind,
                    title=f"Fitness: {np.round(fit, 2)}",
                    **kwargs
                )
            
            grid.append(viz_func)
        
        # Fill remaining slots
        while len(grid) < rows * cols:
            from ariel.ec.drone.inspection.create_subplot import remove_ticks
            grid.append(remove_ticks)
        
        return np.array(grid).reshape(grid_size)
    
    def save_all_plots(
        self,
        output_dir: Union[str, Path],
        prefix: str = "evolution",
        formats: List[str] = ['png', 'pdf']
    ) -> List[str]:
        """
        Save all standard plots for this evolutionary run.
        
        Args:
            output_dir: Directory to save plots
            prefix: Prefix for filenames
            formats: List of file formats
            
        Returns:
            List of saved file paths
        """
        return save_evolution_plots(
            self.population_data,
            self.fitness_data,
            str(output_dir),
            prefix,
            formats
        )
    
    def export_data(self, output_path: Union[str, Path]) -> None:
        """
        Export evolution data to a file.
        
        Args:
            output_path: Path to save data file
        """
        import pickle
        
        data = {
            'population_data': self.population_data,
            'fitness_data': self.fitness_data,
            'parameter_limits': self.parameter_limits,
            'generation_labels': self.generation_labels,
            'summary_report': self.create_summary_report()
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'EvolutionAnalyzer':
        """
        Load evolution data from a file.
        
        Args:
            file_path: Path to data file
            
        Returns:
            EvolutionAnalyzer instance
        """
        import pickle
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        return cls(
            population_data=data['population_data'],
            fitness_data=data['fitness_data'],
            parameter_limits=data.get('parameter_limits'),
            generation_labels=data.get('generation_labels')
        )


class MultiRunAnalyzer:
    """
    Analyzer for comparing multiple evolutionary runs.
    """
    
    def __init__(self, analyzers: List[EvolutionAnalyzer], labels: List[str]):
        """
        Initialize multi-run analyzer.
        
        Args:
            analyzers: List of EvolutionAnalyzer instances
            labels: Labels for each run
        """
        if len(analyzers) != len(labels):
            raise ValueError("Number of analyzers must match number of labels")
        
        self.analyzers = analyzers
        self.labels = labels
        self.n_runs = len(analyzers)
    
    def compare_fitness_evolution(self, figsize: Tuple[float, float] = (14, 8)) -> plt.Figure:
        """
        Create a comparison plot of fitness evolution across runs.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Mean fitness evolution
        for analyzer, label in zip(self.analyzers, self.labels):
            stats = analyzer.fitness_statistics
            generations = np.arange(analyzer.n_generations)
            ax1.plot(generations, stats['mean'], label=label, linewidth=2)
        
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Mean Fitness')
        ax1.set_title('Mean Fitness Evolution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Best fitness evolution
        for analyzer, label in zip(self.analyzers, self.labels):
            stats = analyzer.fitness_statistics
            generations = np.arange(analyzer.n_generations)
            ax2.plot(generations, stats['max'], label=label, linewidth=2)
        
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Best Fitness')
        ax2.set_title('Best Fitness Evolution Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_comparison_summary(self) -> Dict[str, Any]:
        """
        Create a summary comparing all runs.
        
        Returns:
            Dictionary with comparison statistics
        """
        summaries = [analyzer.create_summary_report() for analyzer in self.analyzers]
        
        comparison = {
            'run_labels': self.labels,
            'best_fitness_per_run': [s['fitness_summary']['best_overall'] for s in summaries],
            'final_mean_per_run': [s['fitness_summary']['final_mean'] for s in summaries],
            'improvement_per_run': [s['fitness_summary']['improvement'] for s in summaries],
            'generations_to_best_per_run': [s['fitness_summary']['generations_to_best'] for s in summaries]
        }
        
        # Overall statistics
        best_fitnesses = comparison['best_fitness_per_run']
        comparison['overall_best_run'] = self.labels[np.argmax(best_fitnesses)]
        comparison['overall_best_fitness'] = np.max(best_fitnesses)
        comparison['mean_best_fitness'] = np.mean(best_fitnesses)
        comparison['std_best_fitness'] = np.std(best_fitnesses)
        
        return comparison
    
    def plot_comparison_grid(self, **kwargs) -> plt.Figure:
        """
        Create a grid comparing best individuals from each run.
        
        Returns:
            matplotlib Figure
        """
        from evolution_plotters import plot_evolution_comparison
        
        population_lists = [list(analyzer.population_data) for analyzer in self.analyzers]
        fitness_lists = [list(analyzer.fitness_data) for analyzer in self.analyzers]
        
        fig, axs = plot_evolution_comparison(
            population_lists, fitness_lists, self.labels, **kwargs
        )
        
        return fig


# Convenience functions for quick analysis
def quick_analysis(
    population_data: npt.NDArray,
    fitness_data: npt.NDArray,
    output_dir: Optional[Union[str, Path]] = None,
    create_animation: bool = False
) -> EvolutionAnalyzer:
    """
    Perform a quick analysis of evolutionary data with standard outputs.
    
    Args:
        population_data: Population evolution data
        fitness_data: Fitness evolution data
        output_dir: Directory to save outputs (if None, just returns analyzer)
        create_animation: Whether to create an animation
        
    Returns:
        EvolutionAnalyzer instance
    """
    analyzer = EvolutionAnalyzer(population_data, fitness_data)
    
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save summary report
        report = analyzer.create_summary_report()
        import json
        with open(output_path / "summary_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save plots
        analyzer.save_all_plots(output_path)
        
        # Save fitness evolution plot
        fig = analyzer.plot_fitness_evolution()
        fig.savefig(output_path / "fitness_evolution.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Create animation if requested
        if create_animation:
            try:
                analyzer.create_animation(output_path=str(output_path / "evolution_animation.gif"))
            except Exception as e:
                print(f"Animation creation failed: {e}")
        
        # Export data
        analyzer.export_data(output_path / "evolution_data.pkl")
        
        print(f"Analysis complete. Results saved to {output_path}")
    
    return analyzer


def compare_runs(
    analyzers: List[EvolutionAnalyzer],
    labels: List[str],
    output_dir: Optional[Union[str, Path]] = None
) -> MultiRunAnalyzer:
    """
    Compare multiple evolutionary runs.
    
    Args:
        analyzers: List of EvolutionAnalyzer instances
        labels: Labels for each run
        output_dir: Directory to save comparison outputs
        
    Returns:
        MultiRunAnalyzer instance
    """
    multi_analyzer = MultiRunAnalyzer(analyzers, labels)
    
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save comparison summary
        comparison = multi_analyzer.create_comparison_summary()
        import json
        with open(output_path / "comparison_summary.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Save comparison plots
        fig1 = multi_analyzer.compare_fitness_evolution()
        fig1.savefig(output_path / "fitness_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        fig2 = multi_analyzer.plot_comparison_grid()
        fig2.savefig(output_path / "morphology_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        print(f"Comparison complete. Results saved to {output_path}")
    
    return multi_analyzer


# Example usage
if __name__ == "__main__":
    print("=== Evolution Analyzer Example ===")
    
    # Generate sample data for demonstration
    n_gens, pop_size, n_arms, n_params = 50, 20, 4, 6
    
    # Simulated evolution with improving fitness
    population_data = np.random.rand(n_gens, pop_size, n_arms, n_params)
    fitness_data = np.random.rand(n_gens, pop_size)
    
    # Add fitness improvement trend
    for gen in range(n_gens):
        fitness_data[gen] += gen * 0.01  # Gradual improvement
    
    # Quick analysis
    analyzer = quick_analysis(population_data, fitness_data)
    
    # Print summary
    report = analyzer.create_summary_report()
    print("\nEvolution Summary:")
    print(f"Best fitness: {report['fitness_summary']['best_overall']:.3f}")
    print(f"Improvement: {report['fitness_summary']['improvement']:.3f}")
    print(f"Best found at generation: {report['fitness_summary']['best_generation']}")
    
    print("\n=== Example Complete ===")