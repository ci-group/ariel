"""
Plot learning deltas per individual across generations.

Visualizes how each robot's brain-learning performance improves (or not)
within each generation, and aggregates this across generations.

Usage:
    python plot_learning_deltas_per_individual.py --path __data__/5_minimal_tree_morph_brain_combo_multiprocessing
"""

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console

console = Console()


def load_individuals_with_deltas(db_path: str) -> pd.DataFrame:
    """Load all individuals with learning_deltas from database."""
    conn = sqlite3.connect(db_path)
    query = "SELECT id, time_of_birth, time_of_death, fitness_, tags_ FROM individual"
    df = pd.read_sql(query, conn)
    conn.close()

    # Parse JSON tags and extract learning_deltas
    deltas_list = []
    for _, row in df.iterrows():
        try:
            tags = json.loads(row["tags_"]) if row["tags_"] else {}
            deltas = tags.get("learning_deltas", None)
        except (json.JSONDecodeError, TypeError):
            deltas = None

        deltas_list.append(deltas)

    df["learning_deltas"] = deltas_list
    return df


def plot_individual_delta_trajectories(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot individual learning delta trajectories for each generation."""
    console.log("[cyan]Generating individual delta trajectory plots...[/cyan]")

    # Get min/max generation
    min_gen = int(df["time_of_birth"].min())
    max_gen = int(df["time_of_death"].max())

    # Group by generation
    fig, axes = plt.subplots(max_gen - min_gen + 1, 1, figsize=(12, 3 * (max_gen - min_gen + 1)))
    if max_gen - min_gen == 0:
        axes = [axes]

    for gen_idx, gen in enumerate(range(min_gen, max_gen + 1)):
        ax = axes[gen_idx]

        # Get individuals alive in this generation
        alive_in_gen = df.loc[(df["time_of_birth"] <= gen) & (df["time_of_death"] > gen)]

        # Plot each individual's delta trajectory
        for _, ind in alive_in_gen.iterrows():
            deltas = ind["learning_deltas"]
            if deltas and len(deltas) > 1:  # Skip if no deltas or only init artifact
                # Skip first delta (inf artifact)
                deltas_usable = [d for d in deltas[1:] if np.isfinite(d)]
                if deltas_usable:
                    iterations = range(1, len(deltas_usable) + 1)
                    ax.plot(
                        iterations,
                        deltas_usable,
                        marker="o",
                        label=f"Ind {int(ind['id'])} (fit={ind['fitness_']:.3f})",
                        alpha=0.7,
                    )

        ax.set_title(f"Generation {gen}")
        ax.set_xlabel("Learning Iteration")
        ax.set_ylabel("Delta (Fitness Improvement)")
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "learning_deltas_individual_trajectories.png", dpi=150)
    console.log(f"[green]Saved:[/green] learning_deltas_individual_trajectories.png")
    plt.close()


def plot_aggregate_delta_stats(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot aggregate learning delta statistics per generation."""
    console.log("[cyan]Generating aggregate delta statistics plot...[/cyan]")

    min_gen = int(df["time_of_birth"].min())
    max_gen = int(df["time_of_death"].max())

    gen_means = []
    gen_medians = []
    gen_stds = []
    gen_pos_ratio = []

    for gen in range(min_gen, max_gen + 1):
        alive_in_gen = df.loc[(df["time_of_birth"] <= gen) & (df["time_of_death"] > gen)]

        # Aggregate all usable deltas from all individuals in this generation
        all_deltas = []
        for _, ind in alive_in_gen.iterrows():
            deltas = ind["learning_deltas"]
            if deltas:
                # Skip first delta (inf artifact)
                usable = [d for d in deltas[1:] if np.isfinite(d)]
                all_deltas.extend(usable)

        if all_deltas:
            gen_means.append(np.mean(all_deltas))
            gen_medians.append(np.median(all_deltas))
            gen_stds.append(np.std(all_deltas))
            gen_pos_ratio.append(np.mean([1 if d > 0 else 0 for d in all_deltas]))
        else:
            gen_means.append(np.nan)
            gen_medians.append(np.nan)
            gen_stds.append(np.nan)
            gen_pos_ratio.append(np.nan)

    generations = list(range(min_gen, max_gen + 1))

    # Create figure with twin axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot mean and std
    ax1.plot(
        generations,
        gen_means,
        marker="o",
        label="Mean Delta",
        color="C0",
        linewidth=2,
    )
    ax1.plot(
        generations,
        gen_medians,
        marker="s",
        label="Median Delta",
        color="C1",
        linestyle="--",
        linewidth=2,
    )
    ax1.fill_between(
        generations,
        np.array(gen_means) - np.array(gen_stds),
        np.array(gen_means) + np.array(gen_stds),
        color="C0",
        alpha=0.2,
        label="Mean ± Std",
    )
    ax1.axhline(y=0, color="k", linestyle="-", alpha=0.3, linewidth=1)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Delta (Fitness Improvement)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Twin axis for positive ratio
    ax2 = ax1.twinx()
    ax2.plot(
        generations,
        gen_pos_ratio,
        marker="^",
        label="Fraction Positive",
        color="C2",
        linewidth=2,
    )
    ax2.set_ylabel("Fraction of Positive Deltas", color="C2")
    ax2.tick_params(axis="y", labelcolor="C2")
    ax2.set_ylim([0, 1])
    ax2.legend(loc="upper right")

    fig.suptitle("Learning Delta Statistics per Generation")
    fig.tight_layout()
    fig.savefig(output_dir / "learning_deltas_aggregate_stats.png", dpi=150)
    console.log(f"[green]Saved:[/green] learning_deltas_aggregate_stats.png")
    plt.close()


def plot_heatmap_individual_generations(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot a heatmap of cumulative learning delta per individual per generation."""
    console.log("[cyan]Generating cumulative delta heatmap...[/cyan]")

    min_gen = int(df["time_of_birth"].min())
    max_gen = int(df["time_of_death"].max())

    # Build matrix: rows = individuals, cols = generations
    ind_ids = sorted(df["id"].unique())
    generations = list(range(min_gen, max_gen + 1))

    heatmap_data = np.full((len(ind_ids), len(generations)), np.nan)

    for i, ind_id in enumerate(ind_ids):
        ind_row = df[df["id"] == ind_id].iloc[0]
        birth = int(ind_row["time_of_birth"])
        death = int(ind_row["time_of_death"])

        for j, gen in enumerate(generations):
            if birth <= gen < death:
                # Individual exists in this generation
                deltas = ind_row["learning_deltas"]
                if deltas:
                    # Cumulative delta (skipping first inf artifact)
                    usable = [d for d in deltas[1:] if np.isfinite(d)]
                    if usable:
                        heatmap_data[i, j] = np.sum(usable)

    fig, ax = plt.subplots(figsize=(12, max(4, len(ind_ids) * 0.3)))
    im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(generations)))
    ax.set_xticklabels(generations)
    ax.set_yticks(range(len(ind_ids)))
    ax.set_yticklabels([f"Ind {int(iid)}" for iid in ind_ids])

    ax.set_xlabel("Generation")
    ax.set_ylabel("Individual ID")
    ax.set_title("Cumulative Learning Delta per Individual per Generation\n(Green=improvement, Red=degradation)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Cumulative Delta")

    fig.tight_layout()
    fig.savefig(output_dir / "learning_deltas_heatmap.png", dpi=150)
    console.log(f"[green]Saved:[/green] learning_deltas_heatmap.png")
    plt.close()


def print_summary_stats(df: pd.DataFrame) -> None:
    """Print summary statistics about learning deltas."""
    console.log("[cyan]Summary Statistics:[/cyan]")

    all_deltas = []
    for _, ind in df.iterrows():
        deltas = ind["learning_deltas"]
        if deltas:
            usable = [d for d in deltas[1:] if np.isfinite(d)]
            all_deltas.extend(usable)

    if all_deltas:
        console.log(f"Total delta measurements: {len(all_deltas)}")
        console.log(f"Mean delta: {np.mean(all_deltas):.6f}")
        console.log(f"Median delta: {np.median(all_deltas):.6f}")
        console.log(f"Std delta: {np.std(all_deltas):.6f}")
        console.log(f"Positive deltas: {sum(1 for d in all_deltas if d > 0)} / {len(all_deltas)}")
        console.log(f"Min delta: {np.min(all_deltas):.6f}")
        console.log(f"Max delta: {np.max(all_deltas):.6f}")
    else:
        console.log("[yellow]No learning deltas found in database![/yellow]")


def main():
    parser = argparse.ArgumentParser(description="Plot learning deltas per individual across generations")
    parser.add_argument(
        "--path",
        type=str,
        default="__data__/5_minimal_tree_morph_brain_combo_multiprocessing",
        help="Path to the data directory",
    )
    parser.add_argument(
        "--db-file",
        type=str,
        default=None,
        help="Specific database file (auto-detected if not provided)",
    )
    args = parser.parse_args()

    data_dir = Path(args.path)
    if not data_dir.exists():
        console.log(f"[red]Data directory not found: {data_dir}[/red]")
        return

    # Find database file
    if args.db_file:
        db_path = data_dir / args.db_file
    else:
        db_files = list(data_dir.glob("database_*.db"))
        if not db_files:
            console.log(f"[red]No database files found in {data_dir}[/red]")
            return
        db_path = sorted(db_files)[-1]  # Use most recent

    console.log(f"[cyan]Loading database:[/cyan] {db_path}")

    # Load data
    df = load_individuals_with_deltas(str(db_path))

    # Filter individuals with deltas
    individuals_with_deltas = df[df["learning_deltas"].notna()]
    console.log(f"Individuals with learning deltas: {len(individuals_with_deltas)} / {len(df)}")

    if len(individuals_with_deltas) == 0:
        console.log("[yellow]No individuals with learning deltas found![/yellow]")
        return

    # Create output directory
    output_dir = data_dir / "plots"
    output_dir.mkdir(exist_ok=True)

    # Print summary
    print_summary_stats(individuals_with_deltas)

    # Generate plots
    plot_individual_delta_trajectories(individuals_with_deltas, output_dir)
    plot_aggregate_delta_stats(individuals_with_deltas, output_dir)
    plot_heatmap_individual_generations(individuals_with_deltas, output_dir)

    console.log(f"[green]All plots saved to:[/green] {output_dir}")


if __name__ == "__main__":
    main()
