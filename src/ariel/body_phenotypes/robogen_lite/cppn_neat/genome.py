import random
from collections import deque

from .activations import ACTIVATION_FUNCTIONS, DEFAULT_ACTIVATION
from .connection import Connection
from .node import Node


class Genome:
    """A genome in the NEAT algorithm."""

    def __init__(
        self,
        nodes: dict[int, Node],
        connections: dict[int, Connection],
        fitness: float,
        serialized: dict = None,
    ):
        self.nodes = nodes
        self.connections = connections
        self.fitness = fitness
        self.serialized = serialized
        self._invalidate_cache()

    def _invalidate_cache(self):
        self._input_node_ids = None
        self._output_node_ids = None
        self._incoming_map = None
        self._topo_order = None
        self._is_recurrent = None

    def _build_cache(self):
        self._input_node_ids = [_id for _id, n in self.nodes.items() if n.typ == "input"]
        self._output_node_ids = [_id for _id, n in self.nodes.items() if n.typ == "output"]
        self._incoming_map = {n_id: [] for n_id in self.nodes}
        for conn in self.connections.values():
            if conn.enabled:
                self._incoming_map[conn.out_id].append(conn)
        try:
            self._topo_order = self.get_node_ordering()
            self._is_recurrent = False
        except Exception:
            self._topo_order = None
            self._is_recurrent = True

    def _creates_cycle(self, in_id: int, out_id: int) -> bool:
        """Returns True if adding the edge in_id->out_id would create a cycle."""
        if in_id == out_id:
            return True
        # DFS: can we reach in_id starting from out_id through enabled connections?
        visited: set[int] = set()
        stack = [out_id]
        while stack:
            node = stack.pop()
            if node == in_id:
                return True
            if node in visited:
                continue
            visited.add(node)
            for conn in self.connections.values():
                if conn.enabled and conn.in_id == node:
                    stack.append(conn.out_id)
        return False

    @staticmethod
    def _get_random_weight():
        return random.uniform(-1.0, 1.0)

    @staticmethod
    def _get_random_bias():
        return random.uniform(-1.0, 1.0)

    @staticmethod
    def _get_random_activation():
        """Selects a random activation function from the available list."""
        return random.choice(list(ACTIVATION_FUNCTIONS))

    def copy(self):
        """Returns a new Genome object with identical, deep-copied gene sets."""
        new_nodes = {_id: node.copy() for _id, node in self.nodes.items()}
        new_connections = {
            innov_id: conn.copy() for innov_id, conn in self.connections.items()
        }
        return Genome(new_nodes, new_connections, self.fitness)

    @classmethod
    def random(
        cls,
        num_inputs: int,
        num_outputs: int,
        next_node_id: int,
        next_innov_id: int,
    ):
        """
        Creates a new, randomly initialized Genome with a base topology.
        Initial topology is fully connected inputs to outputs.
        """
        nodes = {}
        connections = {}

        for i in range(num_inputs):
            node = Node(_id=i, typ="input", activation=None, bias=0.0)
            nodes[i] = node

        current_node_id = num_inputs
        for o in range(num_outputs):
            node = Node(
                _id=current_node_id,
                typ="output",
                activation=DEFAULT_ACTIVATION,
                bias=cls._get_random_bias(),
            )
            nodes[current_node_id] = node
            current_node_id += 1

        current_innov_id = next_innov_id
        for in_id in range(num_inputs):
            for out_id in range(num_inputs, num_inputs + num_outputs):
                weight = cls._get_random_weight()
                connection = Connection(
                    in_id,
                    out_id,
                    weight,
                    enabled=True,
                    innov_id=current_innov_id,
                )
                connections[current_innov_id] = connection
                current_innov_id += 1

        return cls(nodes, connections, fitness=0.0)

    def mutate(
        self,
        node_add_rate: float,
        conn_add_rate: float,
        next_innov_id_getter,
        next_node_id_getter,
    ):
        """Applies structural mutation (add_node or add_connection)."""
        if random.random() < conn_add_rate:
            self._mutate_add_connection(next_innov_id_getter)

        if random.random() < node_add_rate:
            self._mutate_add_node(next_innov_id_getter, next_node_id_getter)

    def _mutate_add_connection(self, next_innov_id_getter):
        """Attempts to add a new connection between two existing, non-connected nodes."""
        all_nodes = list(self.nodes.keys())
        if len(all_nodes) < 2:
            return

        in_id, out_id = random.sample(all_nodes, 2)

        # Ensure out_id is not an input (swap if needed)
        if self.nodes[out_id].typ == "input":
            in_id, out_id = out_id, in_id
        # Output nodes cannot be a source
        if self.nodes[in_id].typ == "output":
            return
        # Reject input→input edges
        if self.nodes[in_id].typ == "input" and self.nodes[out_id].typ == "input":
            return
        # Reject edges that would create a cycle
        if self._creates_cycle(in_id, out_id):
            return

        for conn in self.connections.values():
            if conn.in_id == in_id and conn.out_id == out_id:
                return

        new_innov_id = next_innov_id_getter()
        new_weight = self._get_random_weight()
        new_connection = Connection(
            in_id, out_id, new_weight, enabled=True, innov_id=new_innov_id
        )
        self.add_connection(new_connection)

    def _mutate_add_node(self, next_innov_id_getter, next_node_id_getter):
        """Splits an existing connection by inserting a new (hidden) node."""
        if not self.connections:
            return

        conn_to_split: Connection = random.choice(list(self.connections.values()))
        conn_to_split.enabled = False
        self._invalidate_cache()

        new_node_id = next_node_id_getter()
        new_node = Node(
            _id=new_node_id,
            typ="hidden",
            activation=self._get_random_activation(),
            bias=self._get_random_bias(),
        )
        self.add_node(new_node)

        innov_id_1 = next_innov_id_getter()
        conn1 = Connection(
            in_id=conn_to_split.in_id,
            out_id=new_node_id,
            weight=1.0,
            enabled=True,
            innov_id=innov_id_1,
        )
        self.add_connection(conn1)

        innov_id_2 = next_innov_id_getter()
        conn2 = Connection(
            in_id=new_node_id,
            out_id=conn_to_split.out_id,
            weight=conn_to_split.weight,
            enabled=True,
            innov_id=innov_id_2,
        )
        self.add_connection(conn2)

    def crossover(self, other: "Genome", is_maximisation: bool = True) -> "Genome":
        """
        Creates a new offspring Genome by crossing over this Genome (parent A)
        and another Genome (parent B).
        """
        if is_maximisation:
            if self.fitness >= other.fitness:
                fitter_parent = self
                less_fit_parent = other
            else:
                fitter_parent = other
                less_fit_parent = self
        else:
            if self.fitness <= other.fitness:
                fitter_parent = self
                less_fit_parent = other
            else:
                fitter_parent = other
                less_fit_parent = self

        # If fitnesses are equal, the shorter genome should be the less_fit_parent
        if self.fitness == other.fitness and len(self.connections) < len(other.connections):
            fitter_parent = other
            less_fit_parent = self

        offspring_node_genes = {}
        offspring_connection_genes = {}

        all_innov_ids = set(fitter_parent.connections.keys()) | set(
            less_fit_parent.connections.keys()
        )

        for innov_id in all_innov_ids:
            conn_a = fitter_parent.connections.get(innov_id)
            conn_b = less_fit_parent.connections.get(innov_id)

            if conn_a and conn_b:
                chosen_conn = random.choice([conn_a, conn_b])
                offspring_connection_genes[innov_id] = chosen_conn.copy()
            elif conn_a:
                offspring_connection_genes[innov_id] = conn_a.copy()
            elif conn_b:
                if fitter_parent.fitness == less_fit_parent.fitness:
                    offspring_connection_genes[innov_id] = conn_b.copy()

        all_inherited_node_ids = set()
        for conn in offspring_connection_genes.values():
            all_inherited_node_ids.add(conn.in_id)
            all_inherited_node_ids.add(conn.out_id)

        combined_nodes = {**less_fit_parent.nodes, **fitter_parent.nodes}

        for node_id in all_inherited_node_ids:
            node_gene = combined_nodes.get(node_id)
            if node_gene:
                offspring_node_genes[node_id] = node_gene.copy()

        return Genome(offspring_node_genes, offspring_connection_genes, fitness=0.0)

    def add_connection(self, connection: Connection):
        """Adds a connection gene to the genome."""
        if any(
            c.in_id == connection.in_id and c.out_id == connection.out_id
            for c in self.connections.values()
        ):
            raise ValueError("Connection already exists in genome.")
        self.connections[connection.innov_id] = connection
        self._invalidate_cache()

    def add_node(self, node: Node):
        """Adds a node gene to the genome."""
        if node._id in self.nodes:
            raise ValueError("Node already exists in genome.")
        self.nodes[node._id] = node
        self._invalidate_cache()

    def get_node_ordering(self):
        """
        Calculates a topological sort order for feed-forward activation using Kahn's algorithm.
        https://www.geeksforgeeks.org/dsa/topological-sorting-indegree-based-solution/
        """
        graph = {node_id: [] for node_id in self.nodes}
        in_degree = {node_id: 0 for node_id in self.nodes}

        for conn in self.connections.values():
            if conn.enabled:
                graph[conn.in_id].append(conn.out_id)
                in_degree[conn.out_id] += 1

        queue: deque[int] = deque(
            node_id for node_id in self.nodes if in_degree[node_id] == 0
        )
        sorted_order = []

        while queue:
            node_id = queue.popleft()
            sorted_order.append(node_id)

            for neighbor_id in graph[node_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)

        if len(sorted_order) != len(self.nodes):
            raise Exception(
                "A cycle was detected in the genome's graph, cannot create a feed-forward order."
            )

        return sorted_order

    def activate(self, inputs: list[float]) -> list[float]:
        """
        Activates the neural network using cached topology.
        Feed-forward preferred; falls back to iterative relaxation if a cycle exists.
        """
        if self._input_node_ids is None:
            self._build_cache()

        if len(inputs) != len(self._input_node_ids):
            raise ValueError(
                f"Expected {len(self._input_node_ids)} inputs, got {len(inputs)}"
            )

        if not self._is_recurrent:
            node_outputs: dict[int, float] = {}
            for i, node_id in enumerate(self._input_node_ids):
                node_outputs[node_id] = inputs[i]

            for node_id in self._topo_order:
                node = self.nodes[node_id]
                if node.typ == "input":
                    continue
                weighted_sum = sum(
                    node_outputs[conn.in_id] * conn.weight
                    for conn in self._incoming_map[node_id]
                    if conn.in_id in node_outputs
                )
                weighted_sum += node.bias
                node_outputs[node_id] = ACTIVATION_FUNCTIONS[node.activation](
                    weighted_sum
                )

            return [node_outputs[_id] for _id in self._output_node_ids]

        else:
            current_values = {node_id: 0.0 for node_id in self.nodes}
            for i, node_id in enumerate(self._input_node_ids):
                current_values[node_id] = inputs[i]

            n_steps = len(self.nodes) + 1
            for _ in range(n_steps):
                next_values = current_values.copy()
                for node_id, node in self.nodes.items():
                    if node.typ == "input":
                        continue
                    weighted_sum = sum(
                        current_values[conn.in_id] * conn.weight
                        for conn in self._incoming_map[node_id]
                    )
                    weighted_sum += node.bias
                    next_values[node_id] = ACTIVATION_FUNCTIONS[node.activation](
                        weighted_sum
                    )
                current_values = next_values

            return [current_values[_id] for _id in self._output_node_ids]

    def to_dict(self) -> dict:
        """Serializes the Genome to a dictionary."""
        return {
            "nodes": {
                str(k): {
                    "_id": v._id,
                    "typ": v.typ,
                    "activation": v.activation,
                    "bias": v.bias,
                }
                for k, v in self.nodes.items()
            },
            "connections": [
                {
                    "in_id": c.in_id,
                    "out_id": c.out_id,
                    "weight": c.weight,
                    "enabled": c.enabled,
                    "innov_id": c.innov_id,
                }
                for c in self.connections.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict, fitness: float = 0.0) -> "Genome":
        """Reconstructs a Genome object from a dictionary."""
        nodes = {}
        for nid, props in data["nodes"].items():
            nodes[int(nid)] = Node(
                _id=props["_id"],
                typ=props["typ"],
                activation=props["activation"],
                bias=props["bias"],
            )

        connections = {}
        for c in data["connections"]:
            new_conn = Connection(
                in_id=c["in_id"],
                out_id=c["out_id"],
                weight=c["weight"],
                enabled=c["enabled"],
                innov_id=c["innov_id"],
            )
            connections[new_conn.innov_id] = new_conn
        return cls(nodes=nodes, connections=connections, fitness=0.0)
