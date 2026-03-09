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
    validate_genome_dict(a.to_dict())
    validate_genome_dict(b.to_dict())


# ---------------------------------------------------------------------------
# Additional general operators
# ---------------------------------------------------------------------------

def crossover_one_point(a: TreeGenome, b: TreeGenome) -> tuple[TreeGenome, TreeGenome]:
    """Perform one-point crossover on tree genomes.

    Two offspring are produced by exchanging the subtrees rooted at randomly
    chosen non-core nodes from each parent.  The selected subtree is expanded
    to its top ancestor so that an entire branch is swapped.  Returns a pair
    ``(child1, child2)``.  Parents are left unmodified.
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

    root1 = get_top_ancestor(child1, n1)
    root2 = get_top_ancestor(child2, n2)

    subtree_swap(child1, child2, root1, root2)
    # subtree_swap already validates
    return child1, child2


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
