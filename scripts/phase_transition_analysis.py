#!/usr/bin/env python3
"""
Phase Transition and Critical Dynamics Analysis for Spatial EA Experiments

This script analyzes population dynamics from spatial evolutionary algorithm experiments,
focusing on phase transitions between extinction and explosion regimes.

Analyses included:
1. Order parameter computation and critical point estimation
2. Bifurcation diagrams
3. Phase diagrams (2D heatmaps)
4. Survival time distributions (power law analysis)
5. Early warning signals for critical transitions
6. Trajectory clustering

Usage:
    python scripts/phase_transition_analysis.py --experiments-dir __experiments__/experiments_energy/__experiments__
    python scripts/phase_transition_analysis.py --help
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit


def load_grid_summary(experiments_dir: Path) -> pd.DataFrame:
    """Load the grid search summary CSV file."""
    summary_files = list(experiments_dir.glob("grid_search_summary_*.csv"))
    if not summary_files:
        raise FileNotFoundError(f"No grid_search_summary_*.csv found in {experiments_dir}")
    
    # Use the most recent one
    summary_file = sorted(summary_files)[-1]
    print(f"Loading grid summary: {summary_file}")
    return pd.read_csv(summary_file)


def load_experiment_data(experiment_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load aggregated statistics and summary from an experiment directory."""
    stats_file = experiment_dir / "aggregated_statistics.csv"
    summary_file = experiment_dir / "summary.json"
    
    if not stats_file.exists():
        raise FileNotFoundError(f"No aggregated_statistics.csv in {experiment_dir}")
    
    stats_df = pd.read_csv(stats_file)
    
    summary = {}
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
    
    return stats_df, summary


def compute_order_parameter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute order parameter φ = (explosions - extinctions) / total_runs
    
    φ = -1: All extinctions
    φ = 0: Critical point (balanced)
    φ = +1: All explosions
    """
    df = df.copy()
    df['order_parameter'] = (
        df['num_explosions'] - df['num_extinctions']
    ) / df['num_runs']
    
    df['extinction_rate'] = df['num_extinctions'] / df['num_runs']
    df['explosion_rate'] = df['num_explosions'] / df['num_runs']
    
    return df


def estimate_critical_point(df: pd.DataFrame, 
                            parameter_col: str = 'num_mating_zones') -> dict:
    """
    Estimate the critical point where order parameter crosses zero.
    Uses linear interpolation between nearest points.
    """
    # Group by parameter and compute mean order parameter
    grouped = df.groupby(parameter_col)['order_parameter'].mean().reset_index()
    grouped = grouped.sort_values(parameter_col)
    
    params = grouped[parameter_col].values
    order_params = grouped['order_parameter'].values
    
    # Find zero crossing
    sign_changes = np.where(np.diff(np.sign(order_params)))[0]
    
    if len(sign_changes) == 0:
        # No crossing found
        closest_idx = np.argmin(np.abs(order_params))
        return {
            'critical_value': params[closest_idx],
            'method': 'closest_to_zero',
            'order_parameter_at_critical': order_params[closest_idx]
        }
    
    # Linear interpolation at first crossing
    idx = sign_changes[0]
    x1, x2 = params[idx], params[idx + 1]
    y1, y2 = order_params[idx], order_params[idx + 1]
    
    # Solve for x where y = 0
    critical_value = x1 - y1 * (x2 - x1) / (y2 - y1)
    
    return {
        'critical_value': critical_value,
        'method': 'linear_interpolation',
        'bracket': (x1, x2),
        'order_params_at_bracket': (y1, y2)
    }


def plot_order_parameter(df: pd.DataFrame, 
                         parameter_col: str = 'num_mating_zones',
                         output_path: Optional[Path] = None) -> plt.Figure:
    """Plot order parameter vs control parameter with critical point."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by parameter
    grouped = df.groupby(parameter_col).agg({
        'order_parameter': ['mean', 'std'],
        'extinction_rate': 'mean',
        'explosion_rate': 'mean'
    }).reset_index()
    grouped.columns = [parameter_col, 'phi_mean', 'phi_std', 'ext_rate', 'exp_rate']
    
    # Plot order parameter with error bars
    ax.errorbar(grouped[parameter_col], grouped['phi_mean'], 
                yerr=grouped['phi_std'], fmt='o-', capsize=5, 
                label='Order Parameter φ', color='purple', linewidth=2, markersize=8)
    
    # Add critical line
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Critical Point (φ=0)')
    
    # Shade regions
    ax.axhspan(-1, 0, alpha=0.1, color='red', label='Extinction-dominated')
    ax.axhspan(0, 1, alpha=0.1, color='blue', label='Explosion-dominated')
    
    # Estimate and mark critical point
    critical = estimate_critical_point(df, parameter_col)
    ax.axvline(x=critical['critical_value'], color='green', linestyle=':', 
               linewidth=2, label=f"Critical Point ≈ {critical['critical_value']:.1f}")
    
    ax.set_xlabel(parameter_col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Order Parameter φ', fontsize=12)
    ax.set_title('Phase Transition: Order Parameter vs Control Parameter', fontsize=14)
    ax.legend(loc='best')
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_bifurcation_diagram(df: pd.DataFrame,
                              parameter_col: str = 'num_mating_zones',
                              output_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot bifurcation diagram showing population fate vs control parameter.
    Each point represents a run's final state (extinction at 0, explosion at cap).
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for _, row in df.iterrows():
        param_value = row[parameter_col]
        n_ext = int(row['num_extinctions'])
        n_exp = int(row['num_explosions'])
        
        # Jitter for visualization
        jitter_x = np.random.uniform(-0.2, 0.2, max(n_ext, n_exp))
        
        # Plot extinctions at y=0 with jitter
        if n_ext > 0:
            ax.scatter([param_value + j for j in jitter_x[:n_ext]], 
                      np.random.uniform(-2, 2, n_ext),
                      c='red', alpha=0.4, s=30, edgecolors='darkred', linewidth=0.5)
        
        # Plot explosions at y=100 with jitter
        if n_exp > 0:
            ax.scatter([param_value + j for j in jitter_x[:n_exp]], 
                      100 + np.random.uniform(-2, 2, n_exp),
                      c='blue', alpha=0.4, s=30, edgecolors='darkblue', linewidth=0.5)
    
    # Reference lines
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Extinction (N→0)')
    ax.axhline(y=100, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='Explosion (N→cap)')
    
    # Critical point
    critical = estimate_critical_point(df, parameter_col)
    ax.axvline(x=critical['critical_value'], color='green', linestyle=':', 
               linewidth=2, label=f"Critical ≈ {critical['critical_value']:.1f}")
    
    ax.set_xlabel(parameter_col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Final Population State', fontsize=12)
    ax.set_title('Bifurcation Diagram: Bistable Population Dynamics', fontsize=14)
    ax.legend(loc='center right')
    ax.set_ylim(-10, 110)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_phase_diagram_2d(df: pd.DataFrame,
                           x_col: str = 'num_mating_zones',
                           y_col: str = 'mating_energy_amount',
                           output_path: Optional[Path] = None) -> plt.Figure:
    """
    2D phase diagram showing extinction rate across parameter space.
    """
    # Pivot to create 2D grid
    pivot_ext = df.pivot_table(
        index=y_col,
        columns=x_col,
        values='extinction_rate',
        aggfunc='mean'
    )
    
    pivot_exp = df.pivot_table(
        index=y_col,
        columns=x_col,
        values='explosion_rate',
        aggfunc='mean'
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extinction rate
    im1 = axes[0].imshow(pivot_ext.values, cmap='Reds', aspect='auto',
                         extent=[pivot_ext.columns.min()-1, pivot_ext.columns.max()+1,
                                pivot_ext.index.max()+10, pivot_ext.index.min()-10],
                         vmin=0, vmax=1)
    axes[0].set_xlabel(x_col.replace('_', ' ').title())
    axes[0].set_ylabel(y_col.replace('_', ' ').title())
    axes[0].set_title('Extinction Rate')
    plt.colorbar(im1, ax=axes[0], label='Rate')
    
    # Add contour at 0.5
    try:
        cs1 = axes[0].contour(pivot_ext.columns, pivot_ext.index, pivot_ext.values,
                              levels=[0.5], colors='black', linewidths=2)
        axes[0].clabel(cs1, inline=True, fontsize=10, fmt='Critical')
    except:
        pass
    
    # Explosion rate
    im2 = axes[1].imshow(pivot_exp.values, cmap='Blues', aspect='auto',
                         extent=[pivot_exp.columns.min()-1, pivot_exp.columns.max()+1,
                                pivot_exp.index.max()+10, pivot_exp.index.min()-10],
                         vmin=0, vmax=1)
    axes[1].set_xlabel(x_col.replace('_', ' ').title())
    axes[1].set_ylabel(y_col.replace('_', ' ').title())
    axes[1].set_title('Explosion Rate')
    plt.colorbar(im2, ax=axes[1], label='Rate')
    
    # Add contour at 0.5
    try:
        cs2 = axes[1].contour(pivot_exp.columns, pivot_exp.index, pivot_exp.values,
                              levels=[0.5], colors='black', linewidths=2)
        axes[1].clabel(cs2, inline=True, fontsize=10, fmt='Critical')
    except:
        pass
    
    plt.suptitle('Phase Diagram: Parameter Space Exploration', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def plot_survival_time_distribution(df: pd.DataFrame,
                                     output_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot survival time (max generations reached) distribution on log-log axes
    to test for power-law behavior.
    """
    survival_times = df['max_generations_reached'].values
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histogram
    axes[0].hist(survival_times, bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Survival Time (generations)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Survival Time Distribution')
    
    # Log-log survival function (CCDF)
    sorted_times = np.sort(survival_times)
    ccdf = np.arange(len(sorted_times), 0, -1) / len(sorted_times)
    
    axes[1].loglog(sorted_times, ccdf, 'o-', markersize=4)
    axes[1].set_xlabel('Survival Time (generations)')
    axes[1].set_ylabel('P(T > t)')
    axes[1].set_title('Survival Function (Log-Log)')
    axes[1].grid(True, alpha=0.3)
    
    # Fit power law: P(T > t) ~ t^(-alpha)
    try:
        # Use log-linear regression
        log_times = np.log(sorted_times[sorted_times > 0])
        log_ccdf = np.log(ccdf[sorted_times > 0])
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_times, log_ccdf)
        
        # Plot fit
        fit_times = np.linspace(sorted_times.min(), sorted_times.max(), 100)
        fit_ccdf = np.exp(intercept) * fit_times ** slope
        axes[1].loglog(fit_times, fit_ccdf, 'r--', linewidth=2, 
                       label=f'Power law fit: α={-slope:.2f}, R²={r_value**2:.3f}')
        axes[1].legend()
    except Exception as e:
        print(f"Power law fit failed: {e}")
    
    # Semi-log (test for exponential)
    axes[2].semilogy(sorted_times, ccdf, 'o-', markersize=4)
    axes[2].set_xlabel('Survival Time (generations)')
    axes[2].set_ylabel('P(T > t)')
    axes[2].set_title('Survival Function (Semi-Log)')
    axes[2].grid(True, alpha=0.3)
    
    # Fit exponential: P(T > t) ~ exp(-lambda * t)
    try:
        # Linear regression on semi-log
        slope_exp, intercept_exp, r_exp, _, _ = stats.linregress(sorted_times, np.log(ccdf))
        
        fit_ccdf_exp = np.exp(intercept_exp + slope_exp * fit_times)
        axes[2].semilogy(fit_times, fit_ccdf_exp, 'r--', linewidth=2,
                         label=f'Exp fit: λ={-slope_exp:.3f}, R²={r_exp**2:.3f}')
        axes[2].legend()
    except Exception as e:
        print(f"Exponential fit failed: {e}")
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def analyze_early_warnings(stats_df: pd.DataFrame, 
                           window: int = 5) -> dict:
    """
    Compute early warning signals from population time series.
    
    Returns rolling variance and autocorrelation which should increase
    before critical transitions (critical slowing down).
    """
    pop = stats_df['population_mean'].values
    
    # Rolling variance
    variance = pd.Series(pop).rolling(window, min_periods=2).var()
    
    # Rolling autocorrelation (lag-1)
    def autocorr_lag1(x):
        if len(x) < 2:
            return np.nan
        x = np.array(x)
        if np.std(x) == 0:
            return np.nan
        return np.corrcoef(x[:-1], x[1:])[0, 1]
    
    autocorr = pd.Series(pop).rolling(window, min_periods=3).apply(autocorr_lag1)
    
    # Rolling skewness
    skewness = pd.Series(pop).rolling(window, min_periods=3).skew()
    
    return {
        'population': pop,
        'variance': variance.values,
        'autocorrelation': autocorr.values,
        'skewness': skewness.values,
        'generations': stats_df['generation'].values if 'generation' in stats_df.columns else np.arange(len(pop))
    }


def plot_early_warnings(experiments_dir: Path,
                        config_pattern: str = "*num_mating_zones=13*mating_energy_amount=10*",
                        output_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot early warning signals for experiments matching pattern.
    """
    matching_dirs = list(experiments_dir.glob(config_pattern))
    
    if not matching_dirs:
        print(f"No directories matching pattern: {config_pattern}")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for exp_dir in matching_dirs[:1]:  # Use first matching experiment
        try:
            stats_df, summary = load_experiment_data(exp_dir)
            warnings = analyze_early_warnings(stats_df, window=5)
            
            gens = warnings['generations']
            
            # Population
            axes[0, 0].plot(gens, warnings['population'], 'b-', linewidth=1.5)
            axes[0, 0].set_ylabel('Population Mean')
            axes[0, 0].set_title('Population Dynamics')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Variance
            axes[0, 1].plot(gens, warnings['variance'], 'r-', linewidth=1.5)
            axes[0, 1].set_ylabel('Rolling Variance')
            axes[0, 1].set_title('Variance (Early Warning Signal)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Autocorrelation
            axes[1, 0].plot(gens, warnings['autocorrelation'], 'g-', linewidth=1.5)
            axes[1, 0].set_xlabel('Generation')
            axes[1, 0].set_ylabel('Lag-1 Autocorrelation')
            axes[1, 0].set_title('Autocorrelation (Critical Slowing Down)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Skewness
            axes[1, 1].plot(gens, warnings['skewness'], 'm-', linewidth=1.5)
            axes[1, 1].set_xlabel('Generation')
            axes[1, 1].set_ylabel('Rolling Skewness')
            axes[1, 1].set_title('Skewness')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle(f'Early Warning Signals\n{exp_dir.name}', fontsize=12)
            
        except Exception as e:
            print(f"Error processing {exp_dir}: {e}")
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig


def generate_summary_report(df: pd.DataFrame, 
                            experiments_dir: Path,
                            output_path: Optional[Path] = None) -> str:
    """Generate a text summary of the analysis."""
    
    df = compute_order_parameter(df)
    critical = estimate_critical_point(df, 'num_mating_zones')
    
    report = []
    report.append("=" * 60)
    report.append("PHASE TRANSITION ANALYSIS REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Overview
    report.append("OVERVIEW")
    report.append("-" * 40)
    report.append(f"Total configurations: {len(df)}")
    report.append(f"Total runs: {df['num_runs'].sum()}")
    report.append(f"Total extinctions: {df['num_extinctions'].sum()}")
    report.append(f"Total explosions: {df['num_explosions'].sum()}")
    report.append(f"Total completed: {df['num_completed'].sum()}")
    report.append("")
    
    # Critical point
    report.append("CRITICAL POINT ESTIMATION")
    report.append("-" * 40)
    report.append(f"Method: {critical['method']}")
    report.append(f"Critical zone count: {critical['critical_value']:.2f}")
    if 'bracket' in critical:
        report.append(f"Bracket: {critical['bracket']}")
    report.append("")
    
    # Order parameter by zone count
    report.append("ORDER PARAMETER BY ZONE COUNT")
    report.append("-" * 40)
    grouped = df.groupby('num_mating_zones').agg({
        'order_parameter': 'mean',
        'extinction_rate': 'mean',
        'explosion_rate': 'mean'
    }).round(3)
    report.append(grouped.to_string())
    report.append("")
    
    # Best configurations (closest to balance)
    report.append("CONFIGURATIONS CLOSEST TO BALANCE")
    report.append("-" * 40)
    df_sorted = df.reindex(df['order_parameter'].abs().sort_values().index)
    for _, row in df_sorted.head(5).iterrows():
        report.append(f"  Zones={row['num_mating_zones']}, Cost={row['mating_energy_amount']}: "
                     f"φ={row['order_parameter']:.3f} "
                     f"(Ext={row['extinction_rate']:.0%}, Exp={row['explosion_rate']:.0%})")
    report.append("")
    
    # Survival times
    report.append("SURVIVAL TIME STATISTICS")
    report.append("-" * 40)
    report.append(f"Mean max generations: {df['max_generations_reached'].mean():.1f}")
    report.append(f"Std max generations: {df['max_generations_reached'].std():.1f}")
    report.append(f"Min: {df['max_generations_reached'].min()}")
    report.append(f"Max: {df['max_generations_reached'].max()}")
    report.append("")
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"Saved: {output_path}")
    
    return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Phase transition analysis for spatial EA experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with default output directory
  python scripts/phase_transition_analysis.py \\
      --experiments-dir __experiments__/experiments_energy/__experiments__
  
  # Specify output directory
  python scripts/phase_transition_analysis.py \\
      --experiments-dir __experiments__/experiments_energy/__experiments__ \\
      --output-dir __results__/phase_analysis
  
  # Run specific analyses only
  python scripts/phase_transition_analysis.py \\
      --experiments-dir __experiments__/experiments_energy/__experiments__ \\
      --analyses order_parameter bifurcation
        """
    )
    
    parser.add_argument(
        '--experiments-dir', '-e',
        type=Path,
        required=True,
        help='Path to experiments directory containing grid_search_summary_*.csv'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=None,
        help='Output directory for figures and reports (default: <experiments-dir>/phase_analysis)'
    )
    
    parser.add_argument(
        '--analyses', '-a',
        nargs='+',
        choices=['order_parameter', 'bifurcation', 'phase_diagram', 'survival', 'early_warnings', 'all'],
        default=['all'],
        help='Which analyses to run (default: all)'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display plots interactively'
    )
    
    args = parser.parse_args()
    
    # Validate experiments directory
    if not args.experiments_dir.exists():
        raise FileNotFoundError(f"Experiments directory not found: {args.experiments_dir}")
    
    # Set output directory
    output_dir = args.output_dir or args.experiments_dir / "phase_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Determine which analyses to run
    analyses = set(args.analyses)
    if 'all' in analyses:
        analyses = {'order_parameter', 'bifurcation', 'phase_diagram', 'survival', 'early_warnings'}
    
    # Load data
    print("\nLoading data...")
    df = load_grid_summary(args.experiments_dir)
    df = compute_order_parameter(df)
    print(f"Loaded {len(df)} configurations")
    
    # Run analyses
    print("\nRunning analyses...")
    
    if 'order_parameter' in analyses:
        print("  - Order parameter analysis")
        plot_order_parameter(df, output_path=output_dir / "order_parameter.png")
    
    if 'bifurcation' in analyses:
        print("  - Bifurcation diagram")
        plot_bifurcation_diagram(df, output_path=output_dir / "bifurcation_diagram.png")
    
    if 'phase_diagram' in analyses:
        print("  - Phase diagram (2D)")
        if 'mating_energy_amount' in df.columns:
            plot_phase_diagram_2d(df, output_path=output_dir / "phase_diagram_2d.png")
        else:
            print("    Skipping: requires 'mating_energy_amount' column")
    
    if 'survival' in analyses:
        print("  - Survival time distribution")
        plot_survival_time_distribution(df, output_path=output_dir / "survival_distribution.png")
    
    if 'early_warnings' in analyses:
        print("  - Early warning signals")
        plot_early_warnings(args.experiments_dir, 
                           output_path=output_dir / "early_warnings.png")
    
    # Generate summary report
    print("\nGenerating summary report...")
    report = generate_summary_report(df, args.experiments_dir, 
                                     output_path=output_dir / "analysis_report.txt")
    print("\n" + report)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"{'='*60}")
    
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
