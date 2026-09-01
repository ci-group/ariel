r"""Ask whether a finished run produced spatially separated genotypes.

Loads the controllers a run saved, clusters them by genome distance — knowing
nothing about where they are — and then asks whether those clusters occupy
distinct regions of the world. That second step is the emergent-speciation
question the spatial EA was built to test.

The headline number is the **spatial silhouette**: above zero, individuals sit
closer in space to their own genetic cluster than to any other; near zero, the
clusters are spatially mixed however different their genomes are.

Examples
--------
Analyse the most recent run in a results folder::

    python examples/spatial_ea/analyze_clustering.py --results __results__

Pick the metric and the cluster count yourself::

    python examples/spatial_ea/analyze_clustering.py --results __results__ \\
        --distance structural --clusters 4

"""

# Standard library
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Third-party libraries
import matplotlib as mpl
import numpy as np

mpl.use("Agg")

# Local libraries
from ariel import console
from ariel.spatial_ea.clustering import (
    analyze_spatial_clustering,
    cluster_dbscan,
    cluster_hierarchical,
    find_optimal_clusters,
    reduce_dimensions_pca,
)
from ariel.spatial_ea.genotype_distance import (
    compute_genotype_diversity,
    compute_pairwise_distance_matrix,
)
from ariel.spatial_ea.persistence import load_controllers_from_json
from ariel.spatial_ea.visualization import (
    plot_cluster_embedding,
    plot_cluster_quality,
    plot_dendrogram,
    plot_distance_heatmap,
    plot_spatial_clusters,
)

# Global constants
COHERENT_SILHOUETTE = 0.25
MIN_POPULATION = 3


def latest_controller_file(results: Path) -> Path | None:
    """Find the most recent controller export in a folder.

    Parameters
    ----------
    results
        Directory to search.

    Returns
    -------
        The newest ``final_controllers_*.json``, or ``None``.
    """
    candidates = sorted(results.glob("final_controllers_*.json"))
    return candidates[-1] if candidates else None


def main(argv: list[str] | None = None) -> int:
    """Cluster a saved run and report its spatial structure.

    Parameters
    ----------
    argv
        Argument list, defaulting to ``sys.argv[1:]``.

    Returns
    -------
        ``0`` on success, ``1`` when there is nothing usable to analyse.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path.cwd() / "__results__",
    )
    parser.add_argument("--controllers", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.cwd() / "__figures__" / "clustering",
    )
    parser.add_argument(
        "--distance",
        choices=["structural", "weight", "combined"],
        default="combined",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=None,
        help="Cluster count; chosen by silhouette when omitted.",
    )
    parser.add_argument("--world-size", type=float, nargs=2, default=None)
    parser.add_argument("--periodic", action="store_true")
    args = parser.parse_args(argv)

    path = args.controllers or latest_controller_file(args.results)
    if path is None or not path.exists():
        console.print(
            f"[bold red]No controller export found[/] in {args.results}. "
            f"Run the EA with --save-results true first.",
        )
        return 1

    payload = load_controllers_from_json(path)
    controllers = payload.get("controllers", [])
    if len(controllers) < MIN_POPULATION:
        console.print(
            f"[bold red]Only {len(controllers)} controller(s)[/] in {path}; "
            f"need at least {MIN_POPULATION} to cluster.",
        )
        return 1

    genotypes = [c["genotype"] for c in controllers]
    identifiers = [int(c["unique_id"]) for c in controllers]
    positions = np.array([
        c["spawn_position"] if c.get("spawn_position") else [0.0, 0.0, 0.0]
        for c in controllers
    ])

    if args.world_size is not None:
        world_size = (float(args.world_size[0]), float(args.world_size[1]))
    else:
        # Fall back to the extent the population actually occupies.
        span = positions[:, :2].max(axis=0)
        world_size = (float(max(span[0], 1.0)), float(max(span[1], 1.0)))

    console.print(
        f"[bold]Analysing[/] {len(genotypes)} controllers from {path.name}",
    )

    distances = compute_pairwise_distance_matrix(genotypes, args.distance)
    diversity = compute_genotype_diversity(genotypes, args.distance)
    console.print(
        f"  genome diversity: mean {diversity['mean_distance']:.3f}, "
        f"range {diversity['min_distance']:.3f}–"
        f"{diversity['max_distance']:.3f}",
    )

    chosen, scores = find_optimal_clusters(distances, max_clusters=8)
    n_clusters = args.clusters or max(2, chosen)
    console.print(f"  cluster count: {n_clusters}")

    result = cluster_hierarchical(distances, n_clusters=n_clusters)
    density = cluster_dbscan(distances, eps=0.3, min_samples=2)
    console.print(
        f"  hierarchical: {result.n_clusters} clusters, "
        f"silhouette {result.silhouette:.3f}"
        if result.silhouette is not None
        else f"  hierarchical: {result.n_clusters} clusters",
    )
    console.print(
        f"  dbscan: {density.n_clusters} clusters, "
        f"{density.num_noise} unassigned",
    )

    spatial = analyze_spatial_clustering(
        positions,
        result.cluster_labels,
        world_size,
        use_periodic_boundaries=args.periodic,
    )
    silhouette = spatial["spatial_silhouette"]

    console.print("\n[bold]Spatial structure[/]")
    console.print(
        f"  segregation: {spatial['spatial_segregation']:.2f} m "
        f"({spatial['spatial_segregation_normalized']:.1%} of the world "
        f"diagonal)",
    )
    if silhouette is None:
        console.print("  spatial silhouette: not defined for one cluster")
    else:
        verdict = (
            "[bold green]clusters occupy distinct regions[/]"
            if silhouette > COHERENT_SILHOUETTE
            else "[bold yellow]clusters are spatially mixed[/]"
        )
        console.print(f"  spatial silhouette: {silhouette:+.3f} — {verdict}")

    args.output.mkdir(parents=True, exist_ok=True)
    embedding = reduce_dimensions_pca(distances)

    written = [
        plot_spatial_clusters(
            positions,
            result.cluster_labels,
            args.output / "spatial_clusters.png",
            world_size=world_size,
            spatial_silhouette=silhouette,
            cluster_centroids=spatial["cluster_centroids"],
            title_suffix=f"{args.distance} distance, {n_clusters} clusters",
        ),
        plot_cluster_embedding(
            embedding,
            result.cluster_labels,
            args.output / "embedding.png",
            individual_ids=identifiers,
        ),
        plot_distance_heatmap(
            distances,
            args.output / "distance_heatmap.png",
            result.cluster_labels,
            distance_type=args.distance,
        ),
        plot_dendrogram(
            distances,
            args.output / "dendrogram.png",
            labels=[str(i) for i in identifiers],
        ),
    ]
    if scores:
        written.append(
            plot_cluster_quality(
                scores,
                args.output / "cluster_quality.png",
                chosen=n_clusters,
            ),
        )

    console.print(
        f"\n[bold green]Wrote {len(written)} figures to[/] {args.output}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
