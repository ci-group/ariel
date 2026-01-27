import random

from .node import Node
from .connection import Connection
from .activations import ACTIVATION_FUNCTIONS, DEFAULT_ACTIVATION 

class Genome:
    """A genome in the NEAT algorithm."""

    def __init__(self,
                 nodes: dict[int, Node],
                 connections: dict[int, Connection],
                 fitness: float, 
                 serialized: dict = None
                 ):

        self.nodes = nodes
        self.connections = connections
        self.fitness = fitness
        self.serialized = serialized

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
        
        # Deep copy nodes (Node class has a copy method)
        new_nodes = {
            _id: node.copy()
            for _id, node in self.nodes.items()
        }
        
        # Deep copy connections (Connection class has a copy method)
        new_connections = {
            innov_id: conn.copy() 
            for innov_id, conn in self.connections.items()
        }
        
        # Return a new Genome instance
        return Genome(new_nodes, new_connections, self.fitness)

    @classmethod
    def random(cls, 
               num_inputs: int, 
               num_outputs: int, 
               next_node_id: int, 
               next_innov_id: int):
        """
        Creates a new, randomly initialized Genome with a base topology.
        Initial topology is fully connected inputs to outputs.
        """
        
        nodes = {}
        connections = {}
        
        # 1. Create Input Nodes
        for i in range(num_inputs):
            node = Node(_id=i, typ='input', activation=None, bias=0.0)
            nodes[i] = node
            
        # 2. Create Output Nodes (starting ID after inputs)
        current_node_id = num_inputs
        for o in range(num_outputs):
            node = Node(
                _id=current_node_id, 
                typ='output', 
                activation=DEFAULT_ACTIVATION, 
                bias=cls._get_random_bias()
            )
            nodes[current_node_id] = node
            current_node_id += 1
            
        # 3. Create Connections (Fully connect inputs to outputs)
        current_innov_id = next_innov_id
        for in_id in range(num_inputs):
            for out_id in range(num_inputs, num_inputs + num_outputs):
                weight = cls._get_random_weight()
                connection = Connection(in_id, out_id, weight, enabled=True, innov_id=current_innov_id)
                connections[current_innov_id] = connection
                current_innov_id += 1 # Increment for the next unique innovation ID

        return cls(nodes, connections, fitness=0.0)
    

    def mutate(self, 
               node_add_rate: float, 
               conn_add_rate: float, 
               next_innov_id_getter, # function to get/update global innovation ID
               next_node_id_getter   # function to get/update global node ID
               ):
        """
        Applies structural mutation (add_node or add_connection).
        """

        if random.random() < conn_add_rate:
            self._mutate_add_connection(next_innov_id_getter)
        
        if random.random() < node_add_rate:
            self._mutate_add_node(next_innov_id_getter, next_node_id_getter)

    def _mutate_add_connection(self, next_innov_id_getter):
        """Attempts to add a new connection between two existing, non-connected nodes."""
        
        all_nodes = list(self.nodes.keys())
        # We need at least two nodes to form a connection
        if len(all_nodes) < 2:
            return

        # Pick two random distinct nodes
        in_id, out_id = random.sample(all_nodes, 2)

        # (I assume a feed-forward structure)
        if self.nodes[out_id].typ == 'input':
            in_id, out_id = out_id, in_id # Swap to ensure input to non-input

        # Check if connection already exists (using in_id and out_id)
        for conn in self.connections.values():
            if conn.in_id == in_id and conn.out_id == out_id:
                return # Connection already exists

        # Create new connection
        new_innov_id = next_innov_id_getter()
        new_weight = self._get_random_weight()
        new_connection = Connection(in_id, out_id, new_weight, enabled=True, innov_id=new_innov_id)
        
        self.add_connection(new_connection)


    def _mutate_add_node(self, next_innov_id_getter, next_node_id_getter):
        """
        Splits an existing connection by inserting a new (hidden) node.
        """

        if not self.connections:
            return

        # 1. Select a random existing connection to split
        conn_to_split: Connection = random.choice(list(self.connections.values()))
        
        # 2. Disable the old connection
        conn_to_split.enabled = False
        
        # 3. Create the new node
        new_node_id = next_node_id_getter()
        new_node = Node(
            _id=new_node_id, 
            typ='hidden', 
            activation=self._get_random_activation(), 
            bias=self._get_random_bias()
        )
        self.add_node(new_node)
        
        # 4. Create the first new connection (in -> new_node)
        innov_id_1 = next_innov_id_getter()
        conn1 = Connection(
            in_id=conn_to_split.in_id,
            out_id=new_node_id,
            weight=1.0,  # Standard NEAT practice
            enabled=True,
            innov_id=innov_id_1
        )
        self.add_connection(conn1)

        # 5. Create the second new connection (new_node -> out)
        innov_id_2 = next_innov_id_getter()
        conn2 = Connection(
            in_id=new_node_id,
            out_id=conn_to_split.out_id,
            weight=conn_to_split.weight,  # Preserve original weight
            enabled=True,
            innov_id=innov_id_2
        )
        self.add_connection(conn2)

    def crossover(self, other: 'Genome') -> 'Genome':
        """
        Creates a new offspring Genome by crossing over this Genome (parent A) 
        and another Genome (parent B).
        """
        
        # Determine the fitter parent
        if self.fitness >= other.fitness:
            fitter_parent = self
            less_fit_parent = other
        else:
            # make the fitter parent the parent "B"
            fitter_parent = other
            less_fit_parent = self
        
        # If fitnesses are equal, the shorter genome (fewer genes) should be the 'less_fit_parent' 
        # to ensure symmetry in gene inheritance.
        if self.fitness == other.fitness and len(self.connections) < len(other.connections):
             fitter_parent = other
             less_fit_parent = self
        
        offspring_node_genes = {}
        offspring_connection_genes = {}
        
        # 1. Crossover Connection Genes (nothing fancy, just a using the set operator OR here)
        all_innov_ids = set(fitter_parent.connections.keys()) | set(less_fit_parent.connections.keys())
        
        for innov_id in all_innov_ids:
            conn_a = fitter_parent.connections.get(innov_id)
            conn_b = less_fit_parent.connections.get(innov_id)
            
            # We need matching Genes (Innovation ID's)
            if conn_a and conn_b:
                # Inherit randomly 
                chosen_conn = random.choice([conn_a, conn_b])
                
                # Copy the chosen connection
                offspring_connection_genes[innov_id] = chosen_conn.copy() 
                
            # Disjoint/Excess Genes (Innovation ID is only in one parent)
            elif conn_a:
                # Inherit from the Fitter Parent 
                offspring_connection_genes[innov_id] = conn_a.copy()
            
            elif conn_b:
                # Standard NEAT: Only inherit if parents are equally fit.
                if fitter_parent.fitness == less_fit_parent.fitness:
                     offspring_connection_genes[innov_id] = conn_b.copy()
                # Otherwise, skip inheriting the less fit parent's unique gene.


        # 2. Inherit Node Genes
        all_inherited_node_ids = set()
        for conn in offspring_connection_genes.values():
            all_inherited_node_ids.add(conn.in_id)
            all_inherited_node_ids.add(conn.out_id)

        # Get the node gene from the fitter parent if possible, otherwise from the less fit parent
        combined_nodes = {**less_fit_parent.nodes, **fitter_parent.nodes}
        
        for node_id in all_inherited_node_ids:
            # Nodes are inherited without structural change, just copy the properties
            node_gene = combined_nodes.get(node_id)
            if node_gene:
                offspring_node_genes[node_id] = node_gene.copy()
            
        # 3. Create and return the new Genome
        return Genome(offspring_node_genes, offspring_connection_genes, fitness=0.0)


    def add_connection(self, connection: Connection):
        """Adds a connection gene to the genome."""
        if connection not in self.connections.values():
            self.connections[connection.innov_id] = connection
        else:
            raise ValueError("Connection already exists in genome.")
    
    def add_node(self, node: Node):
        """Adds a node gene to the genome."""
        if node not in self.nodes.values():
            self.nodes[node._id] = node
        else:
            raise ValueError("Node already exists in genome.")
        
    def get_node_ordering(self):
        """
        Calculates a topological sort order for feed-forward activation using Kahn's algorithm.
        This ensures a node is evaluated only after all its input nodes are ready.
        https://www.geeksforgeeks.org/dsa/topological-sorting-indegree-based-solution/
        """
        # 1. Build the graph structure and count incoming connections (in-degrees) for each node.
        graph = {node_id: [] for node_id in self.nodes}
        in_degree = {node_id: 0 for node_id in self.nodes}

        for conn in self.connections.values():
            if conn.enabled:
                # An edge goes from the input node to the output node
                graph[conn.in_id].append(conn.out_id)
                # The output node gains an incoming connection
                in_degree[conn.out_id] += 1

        # 2. Initialize a queue with all nodes that have no incoming connections.
        # These are the network's starting points (i.e., the input nodes).
        queue = [node_id for node_id in self.nodes if in_degree[node_id] == 0]
        
        sorted_order = []
        
        # 3. Process nodes in the queue.
        while queue:
            # Dequeue a node that is ready to be evaluated.
            node_id = queue.pop(0)
            sorted_order.append(node_id)

            # For the node we just processed, "remove" its outgoing edges.
            for neighbor_id in graph[node_id]:
                in_degree[neighbor_id] -= 1
                # If a neighbor's in-degree drops to 0, it's now ready to be evaluated.
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)

        # 4. Final check: If the sorted order doesn't include all nodes,
        # this means there was a cycle in the graph (a recurrent connection).
        if len(sorted_order) != len(self.nodes):
            # For a feed-forward CPPN, this indicates an issue.
            raise Exception("A cycle was detected in the genome's graph, cannot create a feed-forward order.")
            
        return sorted_order
    
    def activate(self, inputs: list[float]) -> list[float]:
        """
        Activates the neural network.
        1. Tries a topological sort (Feed-Forward) for speed and precision.
        2. If a cycle is detected, falls back to iterative relaxation (Recurrent).
        """
        
        # 1. Identify Input/Output IDs
        input_node_ids = [_id for _id, node in self.nodes.items() if node.typ == 'input']
        output_node_ids = [_id for _id, node in self.nodes.items() if node.typ == 'output']

        if len(inputs) != len(input_node_ids):
            raise ValueError(f"Expected {len(input_node_ids)} inputs, got {len(inputs)}")

        try:
            # --- STRATEGY A: FEED-FORWARD (Preferred) ---
            ordered_node_ids = self.get_node_ordering()
            
            node_outputs = {}
            
            # Initialize Inputs
            for i, node_id in enumerate(input_node_ids):
                node_outputs[node_id] = inputs[i]

            # Build adjacency cache for speed
            incoming_map = {n_id: [] for n_id in self.nodes}
            for conn in self.connections.values():
                if conn.enabled:
                    incoming_map[conn.out_id].append(conn)

            # Activate in Order
            for node_id in ordered_node_ids:
                node = self.nodes[node_id]
                if node.typ == 'input': continue

                weighted_sum = 0.0
                for conn in incoming_map[node_id]:
                    # In feed-forward, conn.in_id is GUARANTEED to be in node_outputs
                    if conn.in_id in node_outputs:
                        weighted_sum += node_outputs[conn.in_id] * conn.weight
                
                weighted_sum += node.bias
                node_outputs[node_id] = ACTIVATION_FUNCTIONS[node.activation](weighted_sum)

            return [node_outputs[_id] for _id in output_node_ids]

        except Exception:
            # --- STRATEGY B: RECURRENT RELAXATION (Fallback) ---
            # A cycle exists. We update all nodes iteratively for N steps.
            
            # 1. Initialize state (all 0.0)
            current_values = {node_id: 0.0 for node_id in self.nodes}
            
            # 2. Set Inputs
            for i, node_id in enumerate(input_node_ids):
                current_values[node_id] = inputs[i]

            # 3. Pre-calculate adjacency
            incoming_map = {n_id: [] for n_id in self.nodes}
            for conn in self.connections.values():
                if conn.enabled:
                    incoming_map[conn.out_id].append(conn)

            # 4. Iterate (Relaxation)
            # Running for len(nodes) ensures signal can traverse the whole graph
            # regardless of topology.
            n_steps = len(self.nodes) + 1
            
            for _ in range(n_steps):
                next_values = current_values.copy()
                
                for node_id, node in self.nodes.items():
                    if node.typ == 'input': continue
                    
                    weighted_sum = 0.0
                    for conn in incoming_map[node_id]:
                        # Use value from PREVIOUS step
                        weighted_sum += current_values[conn.in_id] * conn.weight
                    
                    weighted_sum += node.bias
                    next_values[node_id] = ACTIVATION_FUNCTIONS[node.activation](weighted_sum)
                
                current_values = next_values

            return [current_values[_id] for _id in output_node_ids]
        
    def to_dict(self) -> dict:
        """Serializes the Genome to a dictionary."""
        return {
            "nodes": {
                str(k): {
                    "_id": v._id, 
                    "typ": v.typ, 
                    "activation": v.activation, 
                    "bias": v.bias
                } 
                # Iterate items() because self.nodes is a dict
                for k, v in self.nodes.items()
            },
            "connections": [
                {
                    "in_id": c.in_id, 
                    "out_id": c.out_id, 
                    "weight": c.weight, 
                    "enabled": c.enabled, 
                    "innov_id": c.innov_id
                }
                # Iterate values() because self.connections is a dict {innov_id: Connection}
                for c in self.connections.values()
            ]
        }

    @classmethod
    def from_dict(cls, data: dict, fitness: float = 0.0) -> 'Genome':
        """Reconstructs a Genome object from a dictionary."""
        # 1. Rebuild Nodes
        nodes = {}
        for nid, props in data["nodes"].items():
            nodes[int(nid)] = Node(
                _id=props["_id"], 
                typ=props["typ"], 
                activation=props["activation"], 
                bias=props["bias"]
            )
            
        # 2. Rebuild Connections 
        connections = {}
        for c in data["connections"]:
            new_conn = Connection(
                in_id=c["in_id"], 
                out_id=c["out_id"], 
                weight=c["weight"], 
                enabled=c["enabled"], 
                innov_id=c["innov_id"]
            )
            connections[new_conn.innov_id] = new_conn
        # 3. Return new Genome instance
        return cls(nodes=nodes, connections=connections, fitness=0.0)