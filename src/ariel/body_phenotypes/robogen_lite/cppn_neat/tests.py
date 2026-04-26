from .genome import Genome
from .id_manager import IdManager


def display_genome(genome: Genome, title: str):
    """Prints a readable summary of a genome's structure."""
    print("-" * 40)
    print(f"🧬 {title}")
    print(f"   Fitness: {genome.fitness}")
    print("   Nodes:")
    # Sort nodes by ID for consistent display order
    for node in sorted(genome.nodes.values(), key=lambda n: n._id):
        act_func = node.activation.__name__ if node.activation else "None"
        print(
            f"     - Node {node._id} (type: {node.typ}, bias: {node.bias:.3f}, act: {act_func})"
        )
    print("   Connections:")
    # Sort connections by innovation ID for consistent display order
    for conn in sorted(genome.connections.values(), key=lambda c: c.innov_id):
        status = "✅ Enabled" if conn.enabled else "❌ Disabled"
        print(
            f"     - Innov {conn.innov_id}: ({conn.in_id} -> {conn.out_id}), w: {conn.weight:.3f} [{status}]"
        )
    print("-" * 40 + "\n")


def test_mutation():
    """Tests the add_node and add_connection mutations."""
    print("=" * 60)
    print("🧪 T E S T I N G   M U T A T I O N 🧪")
    print("=" * 60)

    num_inputs, num_outputs = 2, 1
    initial_innov_id = 2
    num_initial_conns = num_inputs * num_outputs
    next_available_innov_id = initial_innov_id + num_initial_conns

    id_manager = IdManager(
        node_start=num_inputs + num_outputs - 1,
        innov_start=next_available_innov_id - 1,
    )

    genome = Genome.random(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        next_node_id=num_inputs + num_outputs,
        next_innov_id=initial_innov_id,
    )
    display_genome(genome, "Original Genome")

    print("\n>>> Applying mutations...\n")
    genome.mutate(
        node_add_rate=1.0,
        conn_add_rate=1.0,
        next_innov_id_getter=id_manager.get_next_innov_id,
        next_node_id_getter=id_manager.get_next_node_id,
    )

    display_genome(genome, "Mutated Genome (Corrected)")


def test_crossover():
    """Tests the crossover of two parent genomes."""
    print("=" * 60)
    print(" T E S T I N G   C R O S S O V E R ")
    print("=" * 60)

    num_inputs, num_outputs = 2, 1
    initial_innov_id = (
        num_inputs * num_outputs
    ) * 10  # Use a different starting innov to avoid overlap
    num_initial_conns = num_inputs * num_outputs
    next_available_innov_id = initial_innov_id + num_initial_conns

    id_manager = IdManager(
        node_start=num_inputs + num_outputs - 1,
        innov_start=next_available_innov_id - 1,
    )

    parent_a = Genome.random(
        num_inputs, num_outputs, num_inputs + num_outputs, initial_innov_id
    )
    parent_a.mutate(
        1.0, 1.0, id_manager.get_next_innov_id, id_manager.get_next_node_id
    )
    parent_a.fitness = 100.0
    display_genome(parent_a, "Parent A (Fitter)")

    parent_b = Genome.random(
        num_inputs, num_outputs, num_inputs + num_outputs, initial_innov_id
    )
    parent_b.fitness = 50.0
    display_genome(parent_b, "Parent B (Less Fit)")

    print(
        "\n>>> Performing crossover (Fitter Parent A x Less Fit Parent B)...\n"
    )
    offspring = parent_a.crossover(parent_b)
    display_genome(offspring, "Offspring Genome")


# --- Main Execution ---
if __name__ == "__main__":
    test_mutation()
    test_crossover()
