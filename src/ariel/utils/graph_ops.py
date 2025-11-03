"""
Graph operations for robot phenotype manipulation.

Functions for converting between different graph representations.
"""

import json
from typing import Union
import networkx as nx


def robot_json_to_digraph(json_data: Union[dict, str]) -> nx.DiGraph:
    """
    Convert a robot JSON to a NetworkX directed graph.
    
    :param json_data: Robot data as dict or JSON string
    :returns: NetworkX directed graph representation
    """
    if isinstance(json_data, str):
        robot_data = json.loads(json_data)
    else:
        robot_data = json_data
    
    graph = nx.DiGraph()
    
    # Add nodes with their attributes
    for node in robot_data.get("nodes", []):
        node_id = node["id"]
        graph.add_node(node_id, 
                      type=node["type"], 
                      rotation=node["rotation"])
    
    # Add edges with face information
    for edge in robot_data.get("edges", []):
        graph.add_edge(edge["source"], 
                      edge["target"], 
                      face=edge["face"])
    
    return graph


def load_robot_json_file(file_path: str) -> nx.DiGraph:
    """
    Load a robot JSON file and convert it to a NetworkX directed graph.
    
    :param file_path: Path to the JSON file
    :returns: NetworkX directed graph representation
    """
    with open(file_path, 'r') as f:
        robot_data = json.load(f)
    
    return robot_json_to_digraph(robot_data)