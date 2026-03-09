"""Genetic operators for tree genomes.

Simple, conservative operators that preserve core node and face constraints.
"""
from __future__ import annotations

import copy
import random
from typing import Dict, List, Set

import networkx as nx

from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    ModuleType,
    IDX_OF_CORE,
)

from .tree_genome import TreeGenome
from .validation import validate_genome_dict


def add_node(genome: TreeGenome, parent: int, face: str, node_id: int, mtype: str, rotation: str) -> None:
    # add node and edge if allowed; caller should ensure face is free
    genome.nodes[node_id] = {"type": mtype, "rotation": rotation}
    genome.edges.append({"parent": parent, "child": node_id, "face": face})


def remove_subtree(genome: TreeGenome, node_id: int) -> None:
    # remove node and all descendants
    g = genome.to_networkx()
    if node_id not in g:
        return
    descendants = list(nx.descendants(g, node_id))
    descendants.append(node_id)
    for d in descendants:
        genome.nodes.pop(d, None)
    genome.edges = [e for e in genome.edges if e["child"] not in descendants and e["parent"] not in descendants]


def get_top_ancestor(genome: TreeGenome, node_id: int) -> int:
    """Return the highest ancestor of ``node_id`` (stop at core or no parent).

    This walks predecessors in the directed graph until reaching a node whose
    parent is the core (IDX_OF_CORE) or there is no parent. Useful for
    selecting a subtree root that contains a full branch rather than a leaf.
    """
    g = genome.to_networkx()
    current = node_id
    # climb while there is a single parent that is not the core
    while True:
        preds = list(g.predecessors(current))
        if not preds:
            break
        parent = preds[0]
        # stop if parent is core 
        if parent == IDX_OF_CORE:
            break
        current = parent
    return current


def subtree_swap(a: TreeGenome, b: TreeGenome, a_node: int, b_node: int) -> None:
    """Swap subtrees rooted at *a_node* and *b_node* between two genomes.

    The subtrees are detached from their original parents and then inserted into
    the opposite genome.  To avoid identifier collisions we remap all node IDs
    in each subtree before reinsertion.  Only internal edges (both parent and
    child in the subtree) are carried across; any external connections are
    discarded.  After the exchange the two genomes are pruned and validated.
    """
    def extract_subtree(genome: TreeGenome, root: int):
        g = genome.to_networkx()
        subnodes = set([root] + list(nx.descendants(g, root)))
        internal_edges = [e for e in genome.edges if e["parent"] in subnodes and e["child"] in subnodes]
        nodes = {n: genome.nodes[n] for n in subnodes}
        return nodes, internal_edges

    def reassign_ids(nodes: dict[int, dict],
                     edges: list[dict],
                     existing_ids: set[int]):
        """Return copies of *nodes* and *edges* with fresh identifiers.

        IDs are remapped to the smallest integers greater than any value in
        *existing_ids*.  A mapping dict is also returned but only callers that
        need to translate external references will use it.
        """
        if not nodes:
            return {}, [], {}
        mapping: dict[int, int] = {}
        next_id = max(existing_ids, default=-1) + 1
        for old in sorted(nodes):
            mapping[old] = next_id
            next_id += 1
        new_nodes = {mapping[old]: copy.deepcopy(attr) for old, attr in nodes.items()}
        new_edges: list[dict] = []
        for e in edges:
            # preserve any additional keys (face/rotation/etc.)
            edge_copy = {k: copy.deepcopy(v) for k, v in e.items()}
            edge_copy["parent"] = mapping[e["parent"]]
            edge_copy["child"] = mapping[e["child"]]
            new_edges.append(edge_copy)
        return new_nodes, new_edges, mapping

    a_nodes, a_edges = extract_subtree(a, a_node)
    b_nodes, b_edges = extract_subtree(b, b_node)

    # remove the chosen branches from their parents
    remove_subtree(a, a_node)
    remove_subtree(b, b_node)

    # determine occupied ids after removal
    a_existing = set(a.nodes.keys())
    b_existing = set(b.nodes.keys())

    # reassign identifiers before inserting into the other genome
    b_nodes_new, b_edges_new, _ = reassign_ids(b_nodes, b_edges, a_existing)
    a_nodes_new, a_edges_new, _ = reassign_ids(a_nodes, a_edges, b_existing)

    a.nodes.update(b_nodes_new)
    a.edges.extend(b_edges_new)
    b.nodes.update(a_nodes_new)
    b.edges.extend(a_edges_new)

    # clean up any now-invalid connections and perform final checks
    _prune_invalid_edges(a)
    _prune_invalid_edges(b)
    # Note: Validation moved to calling code to allow for temporary invalid states during crossover


# ---------------------------------------------------------------------------
# Additional general operators
# ---------------------------------------------------------------------------

def crossover_subtree(a: TreeGenome, b: TreeGenome) -> tuple[TreeGenome, TreeGenome]:
    """Perform standard GP subtree crossover on tree genomes.

    Two offspring are produced by exchanging randomly chosen subtrees from each parent.
    Unlike one-point crossover on linear genomes, this selects any node (including leaves)
    and swaps the entire subtree rooted at that node. Returns a pair ``(child1, child2)``.
    Parents are left unmodified.
    """
    child1 = copy.deepcopy(a)
    child2 = copy.deepcopy(b)

    def pick_noncore(gen: TreeGenome) -> int | None:
        candidates = [nid for nid in gen.nodes if nid != IDX_OF_CORE]
        return random.choice(candidates) if candidates else None

    n1 = pick_noncore(child1)
    n2 = pick_noncore(child2)
    if n1 is None or n2 is None:
        return child1, child2

    # Standard GP: swap subtrees directly at selected nodes (no ancestor expansion)
    subtree_swap(child1, child2, n1, n2)
    # Validate after crossover
    try:
        validate_genome_dict(child1.to_dict())
        validate_genome_dict(child2.to_dict())
    except ValueError:
        # If validation fails, return copies of parents
        return copy.deepcopy(a), copy.deepcopy(b)
    return child1, child2


# Keep the old function for backward compatibility but mark as deprecated
def crossover_one_point(a: TreeGenome, b: TreeGenome) -> tuple[TreeGenome, TreeGenome]:
    """DEPRECATED: Use crossover_subtree() instead.

    This function incorrectly implements 'one-point crossover' by expanding to top ancestors.
    Standard GP subtree crossover should swap at randomly selected nodes directly.
    """
    import warnings
    warnings.warn(
        "crossover_one_point() is deprecated. Use crossover_subtree() for standard GP behavior.",
        DeprecationWarning,
        stacklevel=2
    )
    return crossover_subtree(a, b)


def mutate_replace_node(genome: TreeGenome) -> None:
    """Point mutation: replace a randomly selected non-core node.

    The node's type and rotation are chosen uniformly at random from the
    allowed options (excluding CORE/NONE).  Any children attached to faces not
    permitted by the new type are deleted along with their subtrees.  The
    genome is validated at the end; invalid mutations are simply no-ops.
    """
    candidates = [nid for nid in genome.nodes if nid != IDX_OF_CORE]
    if not candidates:
        return

    nid = random.choice(candidates)
    old_type = genome.nodes[nid]["type"]

    # choose a new type different from current; exclude CORE/NONE
    types = [t for t in ModuleType if t not in (ModuleType.CORE, ModuleType.NONE)]
    types = [t for t in types if t.name != old_type]
    if not types:
        return
    new_type = random.choice(types).name

    # choose a rotation allowed for the new type
    rotations = [r.name for r in ALLOWED_ROTATIONS[ModuleType[new_type]]]
    new_rot = random.choice(rotations) if rotations else genome.nodes[nid]["rotation"]

    genome.nodes[nid] = {"type": new_type, "rotation": new_rot}

    # drop any children on now-disallowed faces
    allowed = [f.name for f in ALLOWED_FACES[ModuleType[new_type]]]
    for e in list(genome.edges):
        if e["parent"] == nid and e["face"] not in allowed:
            remove_subtree(genome, e["child"])

    # validate and silently ignore issues
    try:
        _prune_invalid_edges(genome)
        validate_genome_dict(genome.to_dict())
    except ValueError:
        pass


def mutate_subtree_replacement(genome: TreeGenome, max_modules: int = 10) -> None:
    """Standard GP subtree mutation: replace a random subtree with a newly generated one.

    Selects a random non-core node, removes its entire subtree, and replaces it
    with a new randomly generated subtree. This is the primary mutation operator
    in canonical GP (Koza, 1992).
    """
    def reassign_ids(nodes: dict[int, dict],
                     edges: list[dict],
                     existing_ids: set[int]):
        """Return copies of *nodes* and *edges* with fresh identifiers."""
        if not nodes:
            return {}, [], {}
        mapping: dict[int, int] = {}
        next_id = max(existing_ids, default=-1) + 1
        for old in sorted(nodes):
            mapping[old] = next_id
            next_id += 1
        new_nodes = {mapping[old]: copy.deepcopy(attr) for old, attr in nodes.items()}
        new_edges: list[dict] = []
        for e in edges:
            edge_copy = {k: copy.deepcopy(v) for k, v in e.items()}
            edge_copy["parent"] = mapping[e["parent"]]
            edge_copy["child"] = mapping[e["child"]]
            new_edges.append(edge_copy)
        return new_nodes, new_edges, mapping

    candidates = [nid for nid in genome.nodes if nid != IDX_OF_CORE]
    if not candidates:
        return

    # Select node to replace
    node_id = random.choice(candidates)

    # Find parent and face connection
    parent_id = None
    parent_face = None
    for e in genome.edges:
        if e["child"] == node_id:
            parent_id = e["parent"]
            parent_face = e["face"]
            break

    # Remove the old subtree
    remove_subtree(genome, node_id)

    # Generate new random subtree and attach it
    if parent_id is not None and parent_face is not None:
        # Create a small random subtree (1-3 nodes typically)
        subtree_size = random.randint(1, min(3, max_modules))
        new_subtree = random_tree(subtree_size)

        # The new subtree has its own core, but we need to graft it onto the parent
        # Remove the core from new_subtree and reattach its children to the parent
        if IDX_OF_CORE in new_subtree.nodes:
            # Get all direct children of the core in the new subtree
            core_children = []
            for e in new_subtree.edges:
                if e["parent"] == IDX_OF_CORE:
                    core_children.append((e["child"], e["face"]))

            # Remove core and its edges
            del new_subtree.nodes[IDX_OF_CORE]
            new_subtree.edges = [e for e in new_subtree.edges if e["parent"] != IDX_OF_CORE]

            # Reassign IDs to avoid conflicts
            existing_ids = set(genome.nodes.keys())
            new_nodes, new_edges, _ = reassign_ids(new_subtree.nodes, new_subtree.edges, existing_ids)

            # Add the new nodes and edges, connecting to the original parent
            genome.nodes.update(new_nodes)
            for child_id, face in core_children:
                if child_id in new_nodes:  # Should always be true after reassignment
                    new_child_id = [k for k, v in new_nodes.items() if v == new_subtree.nodes[child_id]][0]
                    genome.edges.append({"parent": parent_id, "child": new_child_id, "face": face})

            genome.edges.extend(new_edges)

    # Clean up and validate
    _prune_invalid_edges(genome)
    try:
        validate_genome_dict(genome.to_dict())
    except ValueError:
        pass


def mutate_shrink(genome: TreeGenome) -> None:
    """Shrink mutation: replace a node and its subtree with a single new leaf node.

    Selects a random non-core node, removes its entire subtree, and replaces it
    with a single new randomly chosen module type. This reduces tree size and
    is part of standard GP mutation operators.
    """
    candidates = [nid for nid in genome.nodes if nid != IDX_OF_CORE]
    if not candidates:
        return

    node_id = random.choice(candidates)

    # Find parent connection
    parent_id = None
    parent_face = None
    for e in genome.edges:
        if e["child"] == node_id:
            parent_id = e["parent"]
            parent_face = e["face"]
            break

    if parent_id is None:
        return  # Shouldn't happen for valid trees

    # Remove the entire subtree
    remove_subtree(genome, node_id)

    # Replace with a single new node
    types = [t for t in ModuleType if t not in (ModuleType.CORE, ModuleType.NONE)]
    new_type = random.choice(types).name
    rotations = [r.name for r in ALLOWED_ROTATIONS[ModuleType[new_type]]]
    new_rot = random.choice(rotations) if rotations else "DEG_0"

    genome.nodes[node_id] = {"type": new_type, "rotation": new_rot}
    genome.edges.append({"parent": parent_id, "child": node_id, "face": parent_face})

    # Clean up and validate
    _prune_invalid_edges(genome)
    try:
        validate_genome_dict(genome.to_dict())
    except ValueError:
        pass


def mutate_hoist(genome: TreeGenome) -> None:
    """Hoist mutation: replace a parent node with one of its children.

    Selects a random node that has children, then replaces it with one of its
    children, effectively "hoisting" the child up in the tree. This changes
    tree structure without changing size and is part of standard GP operators.
    """
    # Find nodes that have children
    nodes_with_children = set()
    for e in genome.edges:
        nodes_with_children.add(e["parent"])

    candidates = [nid for nid in nodes_with_children if nid != IDX_OF_CORE]
    if not candidates:
        return

    parent_id = random.choice(candidates)

    # Get all children of this parent
    children = [e["child"] for e in genome.edges if e["parent"] == parent_id]
    if not children:
        return

    # Choose one child to hoist
    child_to_hoist = random.choice(children)

    # Find the parent's parent and face
    grandparent_id = None
    parent_face = None
    for e in genome.edges:
        if e["child"] == parent_id:
            grandparent_id = e["parent"]
            parent_face = e["face"]
            break

    if grandparent_id is None:
        return  # Can't hoist root

    # Remove the parent and all its other children (except the one being hoisted)
    for child in children:
        if child != child_to_hoist:
            remove_subtree(genome, child)

    # Remove the parent node and its edges
    if parent_id in genome.nodes:
        del genome.nodes[parent_id]
    genome.edges = [e for e in genome.edges if e["child"] != parent_id and e["parent"] != parent_id]

    # Connect the hoisted child directly to the grandparent
    genome.edges = [e for e in genome.edges if not (e["parent"] == grandparent_id and e["face"] == parent_face)]
    genome.edges.append({"parent": grandparent_id, "child": child_to_hoist, "face": parent_face})

    # Clean up and validate
    _prune_invalid_edges(genome)
    try:
        validate_genome_dict(genome.to_dict())
    except ValueError:
        pass


def random_tree(max_modules: int) -> TreeGenome:
    """Generate a random valid tree genome with up to ``max_modules`` nodes.

    Starts with only the core and iteratively adds a random module on a free
    face until the budget is exhausted or no free faces remain.
    """
    g = TreeGenome()
    g.nodes = {IDX_OF_CORE: {"type": "CORE", "rotation": "DEG_0"}}
    g.edges = []

    next_id = 1
    while next_id <= max_modules:
        # collect available (parent,face) positions
        free = []
        for pid, pdata in g.nodes.items():
            ptype = ModuleType[pdata["type"]]
            allowed_faces = [f.name for f in ALLOWED_FACES[ptype]]
            used = {e["face"] for e in g.edges if e["parent"] == pid}
            for face in allowed_faces:
                if face not in used:
                    free.append((pid, face))
        if not free:
            break
        parent, face = random.choice(free)
        # choose random module type/rotation
        types = [t for t in ModuleType if t not in (ModuleType.CORE, ModuleType.NONE)]
        mtype = random.choice(types).name
        rotations = [r.name for r in ALLOWED_ROTATIONS[ModuleType[mtype]]]
        rot = random.choice(rotations) if rotations else "DEG_0"
        add_node(g, parent, face, next_id, mtype, rot)
        next_id += 1
    return g


def get_tree_depth(genome: TreeGenome) -> int:
    """Calculate the maximum depth from core to any leaf node."""
    if not genome.nodes:
        return 0

    g = genome.to_networkx()
    if len(g) == 0:
        return 0

    max_depth = 0
    stack = [(IDX_OF_CORE, 0)]  # (node, depth)

    while stack:
        node, depth = stack.pop()
        max_depth = max(max_depth, depth)

        for child in g.successors(node):
            stack.append((child, depth + 1))

    return max_depth


def validate_tree_depth(genome: TreeGenome, max_depth: int) -> bool:
    """Check if tree depth is within the specified limit."""
    return get_tree_depth(genome) <= max_depth


def _prune_invalid_edges(genome: TreeGenome) -> None:
    """Remove any edges that violate face constraints or are duplicated.
    
    This is a safety check that filters out edges where the parent type does
    not allow the specified face, removes duplicate parent-face pairs, and
    drops any edges pointing at nonexistent nodes.  Child subtrees flagged for
    deletion are removed only after the scan to avoid modifying the edge list
    during iteration.
    """
    seen = set()  # (parent, face) tuples
    valid_edges: list[dict] = []
    to_prune: list[int] = []

    # iterate over a snapshot of edges to safely modify genome later
    for e in list(genome.edges):
        parent_id = e["parent"]
        child_id = e["child"]
        # drop edges whose parent or child no longer exist
        if parent_id not in genome.nodes or child_id not in genome.nodes:
            continue
        parent_type = ModuleType[genome.nodes[parent_id]["type"]]
        allowed = [f.name for f in ALLOWED_FACES[parent_type]]
        edge_key = (parent_id, e["face"])

        # Check 1: face must be allowed for parent type
        if e["face"] not in allowed:
            to_prune.append(child_id)
            continue

        # Check 2: face must not already be occupied (no duplicates)
        if edge_key in seen:
            to_prune.append(child_id)
            continue

        seen.add(edge_key)
        valid_edges.append(e)

    # remove any subtrees flagged for pruning
    for cid in set(to_prune):
        remove_subtree(genome, cid)

    genome.edges = valid_edges
