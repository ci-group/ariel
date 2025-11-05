#!/usr/bin/env python3
"""
Interactive Evolution Dashboard

This module provides an interactive web dashboard for visualizing evolutionary
computation results using Plotly Dash. It displays morphological analysis
plots from MorphologyAnalyzer with generation selection capabilities.
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import List, Callable, Any

from plotly_morphology_analysis import PlotlyMorphologyAnalyzer
from ariel.ec.a001 import Individual

type Population = List[Individual]


class EvolutionDashboard:
    """Interactive dashboard for evolution visualization."""
    
    def __init__(self, populations: List[Population], decoder: Callable, config: Any):
        """Initialize dashboard with evolution data.
        
        Args:
            populations: List of populations per generation
            decoder: Function to decode Individual genotype to robot graph
            config: Evolution configuration object
        """
        self.populations = populations
        self.decoder = decoder
        self.config = config
        self.analyzer = PlotlyMorphologyAnalyzer()
        
        # Load target robots
        if hasattr(config, 'target_robot_file_path') and config.target_robot_file_path:
            self.analyzer.load_target_robots(str(config.target_robot_file_path))
        
        # Pre-compute fitness timeline
        self._compute_fitness_timeline()
        
        # Cache for computed descriptors per generation
        self._descriptor_cache = {}
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
    
    def _compute_fitness_timeline(self):
        """Compute average fitness for each generation."""
        self.fitness_timeline = []
        
        for gen_idx, population in enumerate(self.populations):
            if population:
                avg_fitness = sum(ind.fitness for ind in population) / len(population)
                self.fitness_timeline.append({
                    'generation': gen_idx,
                    'avg_fitness': avg_fitness,
                    'best_fitness': max(ind.fitness for ind in population),
                    'worst_fitness': min(ind.fitness for ind in population)
                })
    
    def _get_generation_data(self, generation: int):
        """Get or compute morphological data for a specific generation."""
        if generation in self._descriptor_cache:
            return self._descriptor_cache[generation]
        
        if generation >= len(self.populations):
            generation = len(self.populations) - 1
        
        # Load population data into analyzer
        population = self.populations[generation]
        self.analyzer.load_population(population, self.decoder)
        self.analyzer.compute_fitness_scores()
        
        # Cache the results
        self._descriptor_cache[generation] = {
            'descriptors': self.analyzer.descriptors.copy(),
            'fitness_scores': self.analyzer.fitness_scores.copy()
        }
        
        return self._descriptor_cache[generation]
    
    def _setup_layout(self):
        """Setup the dashboard layout."""
        max_generation = len(self.populations) - 1
        
        self.app.layout = html.Div([
            html.H1("Evolution Dashboard", style={'textAlign': 'center', 'marginBottom': 30}),
            
            # Generation control section
            html.Div([
                html.Label("Select Generation:", style={'fontWeight': 'bold', 'marginBottom': 10}),
                dcc.Slider(
                    id='generation-slider',
                    min=0,
                    max=max_generation,
                    step=1,
                    value=max_generation,
                    marks={i: str(i) for i in range(0, max_generation + 1, max(1, max_generation // 10))},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'margin': '20px', 'marginBottom': 40}),
            
            # Fitness timeline (always visible)
            html.Div([
                html.H3("Fitness Over Generations"),
                dcc.Graph(id='fitness-timeline')
            ], style={'margin': '20px', 'marginBottom': 40}),
            
            # Tabbed plots section
            html.Div([
                dcc.Tabs(id='plot-tabs', value='pca-tab', children=[
                    dcc.Tab(label='PCA Analysis', value='pca-tab'),
                    dcc.Tab(label='Fitness Landscapes', value='landscape-tab'),
                    dcc.Tab(label='Fitness Distributions', value='distribution-tab'),
                    dcc.Tab(label='Morphological Diversity', value='diversity-tab'),
                    dcc.Tab(label='Pairwise Features', value='pairwise-tab'),
                ]),
                html.Div(id='tab-content')
            ], style={'margin': '20px'})
        ])
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            Output('fitness-timeline', 'figure'),
            Input('generation-slider', 'value')
        )
        def update_fitness_timeline(selected_generation):
            """Update fitness timeline with highlighted generation."""
            if not self.fitness_timeline:
                return go.Figure()
            
            df = pd.DataFrame(self.fitness_timeline)
            
            fig = go.Figure()
            
            # Add average fitness line
            fig.add_trace(go.Scatter(
                x=df['generation'],
                y=df['avg_fitness'],
                mode='lines+markers',
                name='Average Fitness',
                line=dict(color='blue', width=2)
            ))
            
            # Add best fitness line
            fig.add_trace(go.Scatter(
                x=df['generation'],
                y=df['best_fitness'],
                mode='lines+markers',
                name='Best Fitness',
                line=dict(color='green', width=2)
            ))
            
            # Highlight selected generation
            if selected_generation < len(df):
                selected_row = df.iloc[selected_generation]
                fig.add_trace(go.Scatter(
                    x=[selected_row['generation']],
                    y=[selected_row['avg_fitness']],
                    mode='markers',
                    name=f'Generation {selected_generation}',
                    marker=dict(color='red', size=12, symbol='circle-open', line=dict(width=3))
                ))
            
            fig.update_layout(
                title='Fitness Evolution Over Generations',
                xaxis_title='Generation',
                yaxis_title='Fitness',
                height=400,
                showlegend=True
            )
            
            return fig
        
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('plot-tabs', 'value'),
             Input('generation-slider', 'value')]
        )
        def update_tab_content(active_tab, selected_generation):
            """Update tab content based on selection."""
            # Get data for selected generation
            gen_data = self._get_generation_data(selected_generation)
            
            if active_tab == 'pca-tab':
                return self._create_pca_plot(selected_generation)
            elif active_tab == 'landscape-tab':
                return self._create_landscape_plot(selected_generation)
            elif active_tab == 'distribution-tab':
                return self._create_distribution_plot(selected_generation)
            elif active_tab == 'diversity-tab':
                return self._create_diversity_plot(selected_generation)
            elif active_tab == 'pairwise-tab':
                return self._create_pairwise_plot(selected_generation)
            
            return html.Div("Select a tab to view plots")
    
    def _create_pca_plot(self, generation: int):
        """Create PCA analysis plot using PlotlyMorphologyAnalyzer."""
        self._get_generation_data(generation)  # Load data into analyzer
        fig = self.analyzer.plot_target_descriptors_pca()
        fig.update_layout(title_text=f"PCA Analysis - Generation {generation}")
        return dcc.Graph(figure=fig)
    
    def _create_landscape_plot(self, generation: int):
        """Create fitness landscape plot using PlotlyMorphologyAnalyzer."""
        self._get_generation_data(generation)  # Load data into analyzer
        fig = self.analyzer.plot_fitness_landscapes()
        fig.update_layout(title_text=f"Fitness Landscapes - Generation {generation}")
        return dcc.Graph(figure=fig)
    
    def _create_distribution_plot(self, generation: int):
        """Create fitness distribution plot using PlotlyMorphologyAnalyzer."""
        self._get_generation_data(generation)  # Load data into analyzer
        fig = self.analyzer.plot_fitness_distributions()
        fig.update_layout(title_text=f"Fitness Distributions - Generation {generation}")
        return dcc.Graph(figure=fig)
    
    def _create_diversity_plot(self, generation: int):
        """Create morphological diversity analysis plot using PlotlyMorphologyAnalyzer."""
        self._get_generation_data(generation)  # Load data into analyzer
        fig = self.analyzer.analyze_morphological_diversity()
        fig.update_layout(title_text=f"Morphological Diversity - Generation {generation}")
        return dcc.Graph(figure=fig)
    
    def _create_pairwise_plot(self, generation: int):
        """Create pairwise feature landscape plot using PlotlyMorphologyAnalyzer."""
        self._get_generation_data(generation)  # Load data into analyzer
        fig = self.analyzer.plot_pairwise_feature_landscapes()
        fig.update_layout(title_text=f"Pairwise Feature Landscapes - Generation {generation}")
        return dcc.Graph(figure=fig)
    
    def run(self, host='127.0.0.1', port=8050, debug=True):
        """Run the dashboard server."""
        print(f"Starting Evolution Dashboard at http://{host}:{port}")
        print("Press Ctrl+C to stop the server")
        self.app.run(host=host, port=port, debug=debug)


def run_dashboard(populations: List[Population], decoder: Callable, config: Any, 
                 host='127.0.0.1', port=8050, debug=True):
    """Run the evolution dashboard.
    
    Args:
        populations: List of populations per generation from evolution
        decoder: Function to decode Individual genotype to robot graph  
        config: Evolution configuration object
        host: Server host address
        port: Server port
        debug: Enable debug mode
    """
    dashboard = EvolutionDashboard(populations, decoder, config)
    dashboard.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    # Example usage - this would be called from evolve.py
    print("Evolution Dashboard module - import and call run_dashboard() with your evolution data")