from __future__ import annotations
import json
import ariel.body_phenotypes.robogen_lite.config as config
import contextlib
from collections import deque
import networkx as nx
from collections.abc import Callable
from functools import reduce
import numpy as np
from typing import TYPE_CHECKING

from ariel.ec.genotypes.genotype import Genotype
if TYPE_CHECKING:
    from ariel.ec.crossovers import TreeCrossover
    from ariel.ec.mutations import TreeMutator

SEED = 42
RNG = np.random.default_rng(SEED)

class TreeGenome(Genotype):
    def __init__(self, root: TreeNode | None = None):
        self._root = root

    @staticmethod
    def get_crossover_object() -> TreeCrossover:
        from ariel.ec.crossovers import TreeCrossover
        """Return the crossover operator for tree genomes."""
        return TreeCrossover()
    
    @staticmethod
    def get_mutator_object() -> TreeMutator:
        from ariel.ec.mutations import TreeMutator
        """Return the mutator operator for tree genomes."""
        return TreeMutator()
    
    @staticmethod
    def create_individual(**kwargs: dict) -> TreeGenome:
        """Generate a new TreeGenome individual."""
        return TreeGenome.default_init()
    
    @staticmethod
    def to_digraph(robot_genotype: TreeGenome, use_node_ids: bool = False) -> nx.DiGraph:
        g = nx.DiGraph()
        root = robot_genotype.root
        if root is None:
            return g

        # Stable mapping: either identity (node.id) or contiguous DFS ids.
        if use_node_ids:
            node_key = lambda n: n.id
        else:
            # Assign 0..N-1 in first-seen (DFS) order
            seen: dict[int, int] = {}
            next_id = 0
            def node_key(n: TreeNode) -> int:
                nonlocal next_id
                if n.id not in seen:
                    seen[n.id] = next_id
                    next_id += 1
                return seen[n.id]

        def dfs(parent: TreeNode | None, child: TreeNode, via_face: config.ModuleFaces | None) -> None:
            cid = node_key(child)
            # Add/update child node with attributes (use enum names for JSON-friendliness)
            g.add_node(
                cid,
                type=child.module_type.name,
                rotation=child.rotation.name,
                raw_id=child.id,
            )

            if parent is not None:
                pid = node_key(parent)
                g.add_edge(pid, cid, face=via_face.name if via_face is not None else None)

            # Recurse over children (face -> subnode)
            for face, sub in child.children.items():
                dfs(child, sub, face)

        dfs(None, root, None)
        return g

    @classmethod
    def default_init(cls, *args, **kwargs):
        """Default instantiation with a core root."""
        return cls(root=TreeNode(config.ModuleInstance(type=config.ModuleType.CORE,
                                              rotation=config.ModuleRotationsIdx.DEG_0,
                                              links={})))

    @property
    def root(self) -> TreeNode | None:
        return self._root

    @root.setter
    def root(self, value: TreeNode | None):
        if self._root is not None:
            raise ValueError("Root node cannot be changed once set.")
        self._root = value

    @root.getter
    def root(self) -> TreeNode | None:
        return self._root

    def __repr__(self) -> str:
        """Return a nice string representation of the tree genome."""
        if not self._root:
            return "TreeGenome(empty)"

        node_count = len(list(self._iter_nodes()))
        lines = [f"TreeGenome({node_count} nodes):"]
        lines.extend(self._format_node(self._root, "", True))
        return "\n".join(lines)

    def _iter_nodes(self):
        """Iterator over all nodes in the genome."""
        if self._root:
            yield from self._iter_nodes_recursive(self._root)

    def _iter_nodes_recursive(self, node: TreeNode):
        """Recursively iterate over nodes."""
        yield node
        for child in node.children.values():
            yield from self._iter_nodes_recursive(child)

    def _format_node(self, node: TreeNode, prefix: str, is_last: bool) -> list[str]:
        """Helper method to format a node and its children recursively."""
        connector = "└── " if is_last else "├── "
        node_info = f"{node.module_type.name}({node.rotation.name}, depth={node._depth})"
        lines = [f"{prefix}{connector}{node_info}"]

        child_prefix = prefix + ("    " if is_last else "│   ")

        if node.children:
            child_items = list(node.children.items())
            for i, (face, child) in enumerate(child_items):
                is_last_child = (i == len(child_items) - 1)
                face_connector = "└── " if is_last_child else "├── "
                lines.append(f"{child_prefix}{face_connector}[{face.name}]")

                grandchild_prefix = child_prefix + ("    " if is_last_child else "│   ")
                lines.extend(self._format_node(child, grandchild_prefix, True))

        return lines

    def add_child_to_node(self, node: TreeNode, face: config.ModuleFaces, child_module: config.ModuleInstance):
        """Helper method to add a child to a specific node. However, not recommended to use. Rather use """
        if face not in node.available_faces():
            raise ValueError(f"Face {face} is not available on this node.")

        child_node = TreeNode(child_module, depth=node._depth + 1)
        setattr(node, face.name.lower(), child_node)

    def find_node(self, target_id: int, method: str = "dfs") -> TreeNode | None:
        """Find a node by ID in the entire genome."""
        if not self._root:
            return None

        if method.lower() == "bfs":
            return self._root.find_node_bfs(target_id)
        else:
            return self._root.find_node_dfs(target_id)

    def find_nodes_by_type(self, module_type: config.ModuleType, method: str = "dfs") -> list[TreeNode]:
        """Find all nodes of a specific module type."""
        if not self._root:
            return []

        predicate = lambda node: node.module_type == module_type

        if method.lower() == "bfs":
            return self._root.find_all_nodes_bfs(predicate)
        else:
            return self._root.find_all_nodes_dfs(predicate)

    def copy(self) -> 'TreeGenome':
        """Create a shallow copy of the TreeGenome."""
        new_genome = TreeGenome()
        if self._root:
            new_genome._root = self._root.copy()
        return new_genome

    def __copy__(self) -> 'TreeGenome':
        """Support for copy.copy()."""
        return self.copy()
    
    # ---------- JSON serialization ----------

    @staticmethod
    def from_dict(data: dict) -> "TreeGenome":
        """Deserialize a genome from a dict produced by to_dict()."""
        root_data = data.get("root")
        root = None if root_data is None else TreeNode.from_dict(root_data)
        return TreeGenome(root=root)

    @staticmethod
    def from_json(json_str: str, **kwargs) -> "TreeGenome":
        """Deserialize from a JSON string."""
        return TreeGenome.from_dict(json.loads(json_str))
    
    @staticmethod
    def to_dict(robot_genome: "TreeGenome") -> dict:
        """Serialize the genome into a pure-Python dict (JSON-friendly)."""
        return {
            "root": None if robot_genome._root is None else robot_genome._root.to_dict()
        }

    @staticmethod
    def to_json(robot_genome: "TreeGenome", *, indent: int | None = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(TreeGenome.to_dict(robot_genome), indent=indent)

    # TODO: Implement this
    # def __deepcopy__(self, memo) -> 'TreeGenome':
    #     """Support for copy.deepcopy()."""
    #     new_genome = TreeGenome()
    #     if self._root:
    #         new_genome._root = copy.deepcopy(self._root, memo)
    #     return new_genome


class TreeNode:

    def __init__(self, module: config.ModuleInstance | None = None, depth: int = 0,
                 module_type: config.ModuleType | None = None, module_rotation: config.ModuleRotationsIdx | None = None):
        if module is None:
            assert module_type is not None, "Module type cannot be None if module is not specified"
            assert module_rotation is not None, "Module rotation cannot be None if module is not specified"
            module = config.ModuleInstance(type=module_type, rotation=module_rotation, links={})
        self.module_type = module.type
        self.rotation = module.rotation
        # Keep reference to the original module. Why? Because then the links get automatically filled and we can just read them out when decoding
        self.module = module
        self._depth = depth
        self._front: TreeNode | None = None
        self._back: TreeNode | None = None
        self._right: TreeNode | None = None
        self._left: TreeNode | None = None
        self._top: TreeNode | None = None
        self._bottom: TreeNode | None = None

        self._enable_replacement: bool = False

        self._id = id(self)# if node_id is None else node_id

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, value: int | None):
        raise ValueError("ID cannot be changed once set.")
    
    @classmethod
    def random_tree_node(cls, max_depth: int = 1, branch_prob: float = 0.5) -> 'TreeNode' | None:
        """Create a random tree node with random children up to max_depth."""
        if max_depth < 0:
            return None

        # Exclude CORE and NONE from random selection
        module_type = RNG.choice([mt for mt in config.ModuleType if mt not in {config.ModuleType.CORE, config.ModuleType.NONE}])
        module_rotation = RNG.choice(list(config.ModuleRotationsIdx))
        node = cls(module_type=module_type, module_rotation=module_rotation)

        if max_depth == 0:
            return node

        available_faces = config.ALLOWED_FACES[module_type]
        num_children = RNG.integers(0, len(available_faces) + 1)
        chosen_faces = RNG.choice(available_faces, size=num_children, replace=False)

        for face in chosen_faces:
            if RNG.random() > branch_prob:
                continue  # Skip adding a child based on branch probability
            # Recursively create child nodes with reduced depth
            child_node = cls.random_tree_node(max_depth - 1)
            if child_node is not None:
                node._set_face(face, child_node)

        return node

    @contextlib.contextmanager
    def enable_replacement(self):
        """Context manager to temporarily allow replacement of existing children."""
        all_nodes_to_enable = list(self.get_all_nodes(mode="dfs", exclude_root=False))
        try:
            for n in all_nodes_to_enable:
                n._enable_replacement = True
            yield
        finally:
            for n in all_nodes_to_enable:
                n._enable_replacement = False

    def _can_attach_to_face(self, face: config.ModuleFaces, node: TreeNode | None) -> bool:
        """Check if a node can be attached to the given face."""
        if node is None:
            return True  # Can always detach (set to None)
        if face not in config.ALLOWED_FACES[self.module_type]:
            return False
        # Check if face is already occupied (unless replacement is enabled)
        if not self._enable_replacement:
            face_attr = face.name.lower()
            if getattr(self, f"_{face_attr}") is not None:
                return False  # Face already occupied
        return True

    def _set_face(self, face: config.ModuleFaces, value: 'TreeNode | TreeGenome | None'):
        """Common method to validate and set a face attribute."""
        # Handle TreeGenome by extracting its root
        if isinstance(value, TreeGenome):
            if value.root is None:
                raise ValueError("Cannot attach empty TreeGenome (root is None)")
            actual_value = value.root
        else:
            actual_value = value

        if not self._can_attach_to_face(face, actual_value):
            if actual_value is not None and getattr(self, f"_{face.name.lower()}") is not None:
                raise ValueError(f"{face.name} face already occupied on {self.module_type}")
            raise ValueError(f"Cannot attach to {face.name} face of {self.module_type}")

        # Update the internal attribute with the actual node
        setattr(self, f"_{face.name.lower()}", actual_value)

        # Update the module's links dictionary
        if actual_value is not None:
            self.module.links[face] = self._id
        else:
            self.module.links.pop(face, None)

    def _get_face_given_child(self, child_id: int) -> config.ModuleFaces | None:
        # Weird flex
        # ids_to_faces = reduce(lambda acc, x: {**acc, **x}, map(lambda face_node: {face_node[1].id: face_node[0]}, self.children.items()), {})
        return {face_node[1].id: face_node[0] for face_node in self.children.items()}[child_id]

    def face_mapping(self, face: config.ModuleFaces):
        mapping = {
            config.ModuleFaces.FRONT: self._front,
            config.ModuleFaces.BACK: self._back,
            config.ModuleFaces.RIGHT: self._right,
            config.ModuleFaces.LEFT: self._left,
            config.ModuleFaces.TOP: self._top,
            config.ModuleFaces.BOTTOM: self._bottom,
        }
        return mapping[face]

    @property
    def front(self) -> TreeNode | None:
        return self._front

    @front.setter
    def front(self, value: 'TreeNode | TreeGenome | None'):
        self._set_face(config.ModuleFaces.FRONT, value)

    @property
    def back(self) -> TreeNode | None:
        return self._back

    @back.setter
    def back(self, value: 'TreeNode | TreeGenome | None'):
        self._set_face(config.ModuleFaces.BACK, value)

    @property
    def right(self) -> TreeNode | None:
        return self._right

    @right.setter
    def right(self, value: 'TreeNode | TreeGenome | None'):
        self._set_face(config.ModuleFaces.RIGHT, value)

    @property
    def left(self) -> TreeNode | None:
        return self._left

    @left.setter
    def left(self, value: 'TreeNode | TreeGenome | None'):
        self._set_face(config.ModuleFaces.LEFT, value)

    @property
    def top(self) -> TreeNode | None:
        return self._top

    @top.setter
    def top(self, value: 'TreeNode | TreeGenome | None'):
        self._set_face(config.ModuleFaces.TOP, value)

    @property
    def bottom(self) -> TreeNode | None:
        return self._bottom

    @bottom.setter
    def bottom(self, value: 'TreeNode | TreeGenome | None'):
        self._set_face(config.ModuleFaces.BOTTOM, value)

    @property
    def children(self) -> dict[config.ModuleFaces, TreeNode]:
        result = {}
        for face in config.ALLOWED_FACES[self.module_type]:
            child = self.face_mapping(face)
            if child is not None:
                result[face] = child
        return result

    def __eq__(self, other: object) -> bool:
        """Two nodes are equal if they have the same ID."""
        # This is my approach now (Lukas), we could also check for other equalities.
        if not isinstance(other, TreeNode):
            return False
        return self._id == other._id

    def __hash__(self) -> int:
        """Make TreeNode hashable using its ID."""
        return hash(self._id)

    def __ne__(self, other: object) -> bool:
        """Not equal is the opposite of equal."""
        return not self.__eq__(other)

    def available_faces(self) -> list[config.ModuleFaces]:
        """Return list of faces that can still accept children."""
        available = []
        for face in config.ALLOWED_FACES[self.module_type]:
            if self.face_mapping(face) is None:
                available.append(face)
        return available

    def __repr__(self) -> str:
        """Return a nice string representation of the tree node."""
        child_count = len(self.children)
        available_count = len(self.available_faces())
        child_info = f", {child_count} children" if child_count > 0 else ""
        available_info = f", {available_count} available faces" if available_count > 0 else ""
        return f"TreeNode({self.module_type.name}, {self.rotation.name}, depth={self._depth}{child_info}{available_info})"

    def add_child(self, face: config.ModuleFaces, child_module: config.ModuleInstance):
        """Add a child to the specified face."""
        if face not in self.available_faces():
            raise ValueError(f"Face {face} is not available for attachment.")

        child_node = TreeNode(child_module, depth=self._depth + 1)
        setattr(self, face.name.lower(), child_node)

    def remove_child(self, face: config.ModuleFaces):
        """Remove a child from the specified face."""
        if face not in config.ALLOWED_FACES[self.module_type]:
            raise ValueError(f"Face {face} is not valid for module type {self.module_type}.")

        setattr(self, face.name.lower(), None)

    def get_child(self, face: config.ModuleFaces) -> 'TreeNode | None':
        """Get the child at the specified face."""
        if face not in config.ALLOWED_FACES[self.module_type]:
            return None
        return getattr(self, face.name.lower(), None)

    def find_node_dfs(self, target_id: int) -> 'TreeNode | None':
        """Find a node by ID using Depth-First Search."""
        if self._id == target_id:
            return self

        # Search children recursively
        for child in self.children.values():
            result = child.find_node_dfs(target_id)
            if result is not None:
                return result

        return None

    def find_node_bfs(self, target_id: int) -> 'TreeNode | None':
        """Find a node by ID using Breadth-First Search."""
        queue = deque([self])

        while queue:
            current = queue.popleft()

            if current._id == target_id:
                return current

            # Add all children to queue
            queue.extend(current.children.values())

        return None

    def find_all_nodes_dfs(self, predicate: Callable[TreeNode, bool] | None = None) -> list['TreeNode']:
        """Find all nodes matching a predicate using DFS."""
        result = []

        def dfs_helper(node: 'TreeNode'):
            if predicate is None or predicate(node):
                result.append(node)

            for child in node.children.values():
                dfs_helper(child)

        dfs_helper(self)
        return result

    def find_all_nodes_bfs(self, predicate: Callable[TreeNode, bool] = None) -> list['TreeNode']:
        """Find all nodes matching a predicate using BFS."""
        result = []
        queue = deque([self])

        while queue:
            current = queue.popleft()

            if predicate is None or predicate(current):
                result.append(current)

            queue.extend(current.children.values())

        return result

    def get_all_nodes(self, mode: str = "dfs", exclude_root: bool = True):
        """
        Returns all the nodes in the subtree that has self as root node
        """
        predicate_root: Callable[TreeNode, bool] = (lambda x: x.id != self.id) if exclude_root else (lambda _: True)
        if mode == "dfs":
            return self.find_all_nodes_dfs(predicate=predicate_root)
        elif mode == "bfs":
            return self.find_all_nodes_bfs(predicate=predicate_root)
        else:
            raise ValueError("Invalid mode. Valid modes: dfs, bfs")

    def get_internal_nodes(self, mode: str = "dfs", exclude_root: bool = True):
        """
        Returns all the non-leaf nodes in the subtree that has self as root node
        """
        predicate_root: Callable[TreeNode, bool] = (lambda x: x.id != self.id) if exclude_root else (lambda _: True)
        predicate_internal_nodes: Callable[TreeNode, bool] = (lambda x: len(x.children) > 0)
        predicate_list = [predicate_root, predicate_internal_nodes]
        predicate: Callable[TreeNode, bool] = lambda x: reduce(lambda p, q: p(x) and q(x), predicate_list)
        if mode == "dfs":
            return self.find_all_nodes_dfs(predicate=predicate)
        elif mode == "bfs":
            return self.find_all_nodes_bfs(predicate=predicate)
        else:
            raise ValueError("Invalid mode. Valid modes: dfs, bfs")

    def replace_node(self, node_to_remove: TreeNode, node_to_add: TreeNode):
        """
        1) Finds the parent of node_to_remove in self subtree
        2) Replaces node_to_remove with node_to_add in the parent's children
        3)
        """
        predicate_is_parent = lambda x: node_to_remove in set(x.children.values())
        parent = self.find_all_nodes_dfs(predicate=predicate_is_parent)
        # print("PARENTS LENGTH",len(parent))
        if not parent or len(parent) > 1:
            raise RuntimeError("Father not found, are you sure node_to_remove is in subtree?")
        # We expect a list of len 1 in which there is the parent
        parent = parent[0]
        parent._set_face(parent._get_face_given_child(node_to_remove.id), node_to_add)

    def copy(self) -> 'TreeNode':
        """Create a deep copy of this node and all its children."""
        # Create new module instance
        new_module = config.ModuleInstance(
            type=self.module_type,
            rotation=self.rotation,
            links={}  # Will be rebuilt
        )

        # Create new node
        new_node = TreeNode(new_module, depth=self._depth,)

        # Recursively copy children
        for face, child in self.children.items():
            child_copy = child.copy()
            new_node._set_face(face, child_copy)

        return new_node

    def __copy__(self) -> 'TreeNode':
        """Support for copy.copy() - creates deep copy for tree structures."""
        return self.copy()
    
    # ---------- JSON / dict serialization ----------
    def to_dict(self) -> dict:
        """
        Serialize this node recursively.
        Enums are stored by name (string). Children are a mapping face->child.
        """
        # Children as {face_name: child_dict}
        children_dict = {
            face.name: child.to_dict()
            for face, child in self.children.items()
        }
        return {
            "id": self._id,
            "depth": self._depth,
            "module_type": self.module_type.name,
            "rotation": self.rotation.name,
            "children": children_dict,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TreeNode":
        """
        Rebuild a TreeNode (and its subtree) from dict.
        Uses _set_face so that ModuleInstance.links remains consistent.
        """
        #node_id = data["id"]
        depth = data["depth"]
        module_type = config.ModuleType[data["module_type"]]
        rotation = config.ModuleRotationsIdx[data["rotation"]]

        # Create the node with a fresh ModuleInstance and the preserved id/depth
        node = cls(
            module=config.ModuleInstance(type=module_type, rotation=rotation, links={}),
            depth=depth,
            #node_id=node_id,
        )

        # Rebuild children through _set_face so links are updated properly.
        children = data.get("children", {}) or {}
        for face_name, child_payload in children.items():
            face = config.ModuleFaces[face_name]
            child_node = cls.from_dict(child_payload)
            node._set_face(face, child_node)

        return node



def test():
    genome = TreeGenome()
    genome.root = TreeNode(config.ModuleInstance(type=config.ModuleType.BRICK, rotation=config.ModuleRotationsIdx.DEG_90, links={}))
    genome.root.front = TreeNode(config.ModuleInstance(type=config.ModuleType.BRICK, rotation=config.ModuleRotationsIdx.DEG_45, links={}))
    genome.root.left = TreeNode(config.ModuleInstance(type=config.ModuleType.BRICK, rotation=config.ModuleRotationsIdx.DEG_45, links={}))
    #genome.root.front = TreeNode(config.ModuleInstance(type=config.ModuleType.HINGE, rotation=config.ModuleRotationsIdx.DEG_90, links={}))
    with genome.root.enable_replacement():
        genome.root.front = TreeNode(config.ModuleInstance(type=config.ModuleType.HINGE, rotation=config.ModuleRotationsIdx.DEG_45, links={}))
    print(genome.root.front.available_faces())
    print(genome)  # Shows full tree structure
    print(genome.root)  # Shows node details with available faces
    with genome.root.enable_replacement():
        genome.root.replace_node(genome.root.front, genome.root.left)
    print(genome.root.get_all_nodes("dfs", True))


if __name__ == "__main__":
    test()