import mujoco as mj
import numpy as np
import numpy.typing as npt
from typing import TYPE_CHECKING, Any, Sequence

# Local libraries
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.simulation.environments import OlympicArena

def find_in_out_size(
    robot_graph : Any, 
    spawn_pos : Sequence[float]
    ) -> Sequence[int]:
    """Finds out the input and output sizes, based on a robot specification

    Args:
        robot_graph (Any): The robot graph json file
        SPAWN_POS (List[float...]): Must be the global variable for Spawning position
                                    to ensure no undefined behaviors.

    Returns:
        int: Required input size for NN controller
        int: Required output size for NN controller
    """
    core_test = construct_mjspec_from_graph(robot_graph)
    
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world_test = OlympicArena()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world_test.spawn(core_test.spec, spawn_position=spawn_pos)

    # Generate the model and data
    model_test = world_test.spec.compile()
    data_test = mj.MjData(model_test)
    
    # Extract input and output sizes
    input_size = len(data_test.qpos) + len(data_test.qvel)
    output_size = model_test.nu
    
    return (input_size, output_size)

def compute_brain_genome_size(network_specs):
    """Compute the brain_genome size for a given brain network specification. Assumes
       a neural network with at least one hidden layer.

    Args:
        network_specs (dict): A dictionary containing the following keys: "input_size" "output_size" "hidden_size" "no_hidden_layers"

    Returns:
        out : The number of weights needed for a single individual.
    """
    input_size =        network_specs["input_size"]
    output_size =       network_specs["output_size"]
    hidden_size =       network_specs["hidden_size"]
    no_hidden_layers =  network_specs["no_hidden_layers"]
    return input_size * hidden_size + (no_hidden_layers - 1) * (hidden_size**2)  + hidden_size * output_size 

def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
    matrices,
    sim_config = None
    ) -> npt.NDArray[np.float64]:
    """ Feedforward neural network controller function

    Args:
        model (mj.MjModel): MjModel
        data (mj.MjData): MjData
        matrices (list): A list of weight matrices for the network

    Returns:
        npt.NDArray[np.float64]: Control outputs to hinges
    """
    ## Get inputs, in this case the positions of the actuator motors (hinges)
    inputs = np.concatenate([data.qpos, data.qvel])

    ## Run the inputs through all layers of the network.
    for W in matrices[:-1]:
        inputs = np.tanh(np.dot(inputs, W))
    outputs = np.tanh(np.dot(inputs, matrices[-1]))

    ## Scale the outputs
    return outputs * np.pi

def decode_brain_genotype(brain_genotype, network_specs):
    """Decode the brain genotype representation into weight matrices

    Args:
        brain_genotype (list): The genotype representation
        network_specs (dict): A dict of network specifications, including the following keys - 'input_size' 'output_size' 'hidden_size' 'no_hidden_layers'

    Returns:
        list: A list of network matrices
    """
    ## Get the network specifications 
    input_size =        network_specs["input_size"]
    output_size =       network_specs["output_size"]
    hidden_size =       network_specs["hidden_size"]
    no_hidden_layers =  network_specs["no_hidden_layers"]
    
    ## Preparation for decoding
    idx = 0
    matrices = []
    n_params = [input_size * hidden_size, hidden_size**2, hidden_size * output_size]
    
    # Input - Hidden matrix
    matrices.append(np.array(brain_genotype[idx:idx + n_params[0]]).reshape((input_size, hidden_size)))
    idx += n_params[0]
    # Hidden - Hidden matrices
    for _ in range(no_hidden_layers - 1):
        matrices.append(np.array(brain_genotype[idx:idx + n_params[1]]).reshape((hidden_size, hidden_size)))
        idx += n_params[1]
    # Hidden - Output matrix
    matrices.append(np.array(brain_genotype[idx:idx+ n_params[2]]).reshape((hidden_size, output_size)))
    
    return matrices

def decode_body_genotype(genotype, genotype_size):
    idx = 0
    out = []
    for _ in range(len(genotype)//genotype_size):
        gene = np.array(genotype[idx:idx + genotype_size]).astype(np.float32)
        out.append(gene)
        idx += genotype_size
    return out
