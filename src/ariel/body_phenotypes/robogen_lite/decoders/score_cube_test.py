"""
Test script for MorphologyDecoderCubePruning.
Diagnoses why the cube decoder is producing only single-module robots.
"""
from rich.console import Console
from ariel.body_phenotypes.robogen_lite.cppn_neat.genome import Genome
from ariel.body_phenotypes.robogen_lite.decoders.score_cube import (
    MorphologyDecoderCubePruning,
)

console = Console()

# Create a simple random CPPN genome
console.log("[bold]Creating random CPPN genome...[/bold]")
genome = Genome.random(
    num_inputs=6,
    num_outputs=9,  # 1 (connection) + 4 (types) + 4 (rotations)
    next_node_id=15,
    next_innov_id=24,
)

console.log(f"Genome nodes: {len(genome.nodes)}, connections: {len(genome.connections)}")

# Test the decoder
console.log("\n[bold]Running cube pruning decoder...[/bold]")
decoder = MorphologyDecoderCubePruning(cppn_genome=genome, max_modules=15)
robot_graph = decoder.decode()

console.log(f"\n[green]Decoded robot:[/green]")
console.log(f"  Modules: {robot_graph.number_of_nodes()}")
console.log(f"  Edges: {robot_graph.number_of_edges()}")
console.log(f"  Cube size: {len(decoder.score_cube)}")

# Inspect the score cube
console.log(f"\n[bold]Score cube analysis:[/bold]")
if decoder.score_cube:
    scores = [entry["score"] for entry in decoder.score_cube.values()]
    console.log(f"  Min score: {min(scores):.4f}")
    console.log(f"  Max score: {max(scores):.4f}")
    console.log(f"  Mean score: {sum(scores) / len(scores):.4f}")
    
    # Count high-scoring entries
    high_score_count = sum(1 for s in scores if s > 0)
    console.log(f"  High-scoring (>0): {high_score_count}/{len(scores)}")
else:
    console.log("  [red]Cube is empty![/red]")
