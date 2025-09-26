import torch
from _typeshed import Incomplete
from torch.fx.experimental.partitioner_utils import Device as Device, NodeLatency as NodeLatency, Partition as Partition, PartitionMode as PartitionMode, PartitionerConfig as PartitionerConfig, get_extra_size_of as get_extra_size_of, get_latency_of_partitioned_graph as get_latency_of_partitioned_graph, get_partition_to_latency_mapping as get_partition_to_latency_mapping
from torch.fx.graph_module import GraphModule as GraphModule
from torch.fx.node import Node as Node, map_arg as map_arg
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes as get_size_of_all_nodes
from torch.fx.passes.split_module import split_module as split_module
from typing import NamedTuple

class DAGNode:
    """DAGNode class maintains useful information for a partition (submodule),
    and its input submodules and output submodules.
    """
    submodule_node: Node
    input_nodes: list[Node]
    output_nodes: list[Node]
    logical_device_ids: list[int]
    size_bytes: Incomplete
    def __init__(self, submodule_node: Node, input_nodes: list[Node], output_nodes: list[Node], logical_device_ids: list[int], size_bytes: int) -> None: ...
    def __str__(self) -> str: ...

class DAG:
    """DAG class contains all the DAG nodes"""
    nodes: list[DAGNode]
    def __init__(self) -> None: ...
    def create_node(self, submodule_node: Node, input_nodes: list[Node], output_nodes: list[Node], logical_devices: list[int], size_bytes: int) -> None: ...

class PartitionResult(NamedTuple):
    """NameTuple used for returning DAG and a new fx module"""
    dag: DAG
    module_with_submodules: GraphModule

def reset_partition_device(partitions) -> None: ...
def combine_two_partitions(partition_0: Partition, partition_1: Partition, partitions: list[Partition]) -> None:
    """Given a list of partitions and its two partitions,
    combine these two partitions into a new one appending to the partitions
    and remove the previous two partitions from the list of partitions
    """
def set_parents_and_children(partitions: list[Partition]) -> None:
    """Given a list of partitions, mark parents and children for each partition"""
def reorganize_partitions(partitions: list[Partition]) -> None:
    """Given a list of partitions, reorganize partition id,
    its parents and its children for each partition
    """
def get_bfs_level_partition(partitions: list[Partition]) -> None:
    """Given a list of partitions,
    mark the bfs level for each partition
    """
def get_node_to_partition_mapping(partitions: list[Partition]) -> dict[Node, int]:
    """Given a list of partitions,return node to partition mapping"""
def get_logical_id_to_device(devices: list[Device]) -> dict[int, Device]:
    """Get a mapping from device logical ID to Device object."""
def get_device_partition_stats(partitions: list[Partition], devices: list[Device]) -> tuple[dict[Device, list[Partition]], dict[Device, int], list[Partition]]:
    """Given a list of partitions and a list of devices, returns:
    1. A mapping from device to partitions on it;
    2. A mapping from device to its remaining memory size;
    3. A list of partitions that do not have a device.
    """
def get_device_to_partitions_mapping(partitions: list[Partition], devices: list[Device]):
    """Given a list of partitions and a list of devices,
    map each partition into a device.
    """
def check_dependency(partition):
    """Given a partition,check if there is a circular dependency on
    this partition using bfs
    """

class Partitioner:
    """A fx module may not fit into one device.
    Partitioner class helps partition one fx module into submodules (partitions),
    so that the submodules can be executed crossing different accelerators.
    The main function of this class is self.partition_graph.
    It partitions the fx module based on the scheme specified in partition_config
    A DAG structure is returned
    along with a new fx module with submodule nodes.
    """
    partitions: list[Partition]
    node_to_partition: dict[Node, int]
    devices: list[Device]
    def __init__(self) -> None: ...
    graph_module: Incomplete
    torch_module: Incomplete
    def partition_graph(self, fx_module: GraphModule, torch_module: torch.nn.Module, partitioner_config: PartitionerConfig) -> PartitionResult:
        """Given the fx module, torch module and partitioner_config,
        find the partitions, do the partitions,
        and then return a DAG and a new fx module with submodule nodes (partitions)
        """
    def find_single_partition(self, total_size_of_graph, logical_device_id: int = 0) -> None:
        """Fit the whole fx module into one device"""
    def size_based_partition(self) -> None:
        """This method is to partition the fx module based on memory size.
        It uses greedy approach. The result may not be the best.
        The basic idea is:
        Step 1:
        Find a device which has enough memory to fit the current node, create a empty partition
        with the size of that device.
        Then keep adding the following nodes into the partition until the partition is full.
        Step 2:
        Repeat Step 1 until no device left
        Step 3:
        If some nodes are left, create a partition for each left node (single node partition).
        and then try to map those partitions into logical devices with enough mem left.
        """
    def saturate_host(self) -> None:
        """Saturate host by assigning replicates to unused devices with enough memory.
        It uses a greedy approach to find a next available set of devices to place all split
        partitions: For each used device, it searches for an idle device with minimal memory
        size that can hold all the partition located on that device; If the search is successful
        for all used devices, it then assigns the new devices' logical ID to the corresponding
        partition.
        """
    def do_partition(self) -> GraphModule:
        """Return a new fx module with submodule nodes (partitions)."""
    def dump_dag(self, module_with_submodules: GraphModule) -> DAG:
        """Return the dag structure and the new fx module with submodules."""
    def create_partition(self) -> Partition:
        """Create a partition and append it to self.partitions."""
    def create_single_node_partition(self, node) -> None:
        """Create a partition for a single node"""
    def sparse_nn_partition(self, available_mem_bytes: int) -> None:
        """This method partition a sparse nn module.
        It is size based partition but different from size_based_partition,
        it only works when all the devices have same memory size (available_mem_bytes).
        In the future, devices with different mem sizes will be supported like size_based_partition.
        It first traverse all the nodes and do the partitions based on the same memory size.
        If the current partition has no enough memory left for a new op node
        (call_module, call_method, call_function), a new partition is created.
        When crossing the boundary between non-embedding nodes and embedding nodes,
        a new partition is created regardlessly.
        For example, if the current node is a non-embedding node but the next node is an
        embedding node, a new partition is created for the next node.
        After the partition, the partitions are combined as much as possible.
        The rule is that a non-embedding partition only
        combines with another non-embedding one.
        So as the embedding partitions.
        """
    def cost_aware_partition(self, transfer_rate_bytes_per_sec: float, node_to_latency_mapping: dict[Node, NodeLatency]) -> None:
        """This method is to partition the fx module based on the cost.
        The cost is the total latency of running the whole fx module.
        In partitioner_utils.py, the cost model is built.
        The cost aware partition algorithm is:
        #1. At every beginning, each node is a partition.
            Then we map all the partitions to the devices
            and calculate the cost
        #2. Then try to pre-combine any two of the partitions if the two
            partitions can be combined.
            (the bfs level is less than 2 or two partitions are connected and
            can find partition to device mapping)
            See if any partition pair could reduce the current cost.
            Choose the pair that shows the minimum cost and then combine them
        #3. Repeat #2 until the cost cannot be reduced.
        """
    def kl_based_partition(self, transfer_rate_bytes_per_sec: float, node_to_latency_mapping: dict[Node, NodeLatency]) -> None:
        """This function is a cost aware partition based
        on Kernighan-Lin algorithm.
        First, the graph is partitioned using size_based_partition.
        Then, each node is swapped with any other node in a different
        partition, and at the same time, the cost is estimated after
        the swapping.
        For example, we have nodes n0, n1, n2, n3 and n4.
        Using size_based_partition, n0 and n1 are in Partition p0.
        n2, n3 and n4 in Partition p1. The current cost is estimated.
        We first tried using n0 to swap with n2 from the other partition.
        Then we see that swapping n0 and n2 shows a lower cost
        than the current cost and it is the minimum among other pairs like
        (n0, None)(This means moving n0 to Partition without swapping other nodes),
        (n0, n3) and (n0, n4). We swap n0 and n2 and set the new cost
        as the current cost.
        Then We repeat this process for all the other nodes until all swapping pairs
        are tried.
        """
    def aot_based_partition(self, node_to_partition_mapping, partition_to_logical_device_mapping) -> None:
        """This function helps to rebuild the partitions given the nodes and its
        corresponding partition id
        """
