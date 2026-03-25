"""
Robot Evolution Example using TreeGenome Morphologies with Neural Network Controllers

This script demonstrates joint-evolution of robot morphologies and controllers:
- **Morphologies**: Represented using TreeGenome (fast genetic programming approach)
- **Controllers**: Neural networks with phase inputs trained via CMA-ES learning

The example evolves morphologically diverse robots to move towards a target position.
"""

# Imports
import contextlib
import copy
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco as mj
import nevergrad as ng
import numpy as np
import torch
from mujoco import viewer
from rich.console import Console
from rich.progress import track
from rich.traceback import install
from torch import nn

# Ariel Imports
from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.ec import EA, EAOperation, EASettings, Individual, Population
from ariel.ec import config as engine_config
from ariel.ec.genotypes.tree.operators import (
    _prune_invalid_edges,
    crossover_subtree,
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
    validate_tree_depth,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict
from ariel.simulation.controllers.controller import Controller, Tracker
from ariel.simulation.controllers.utils.data_get import get_state_from_data
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.runners import simple_runner

if TYPE_CHECKING:
    from networkx import DiGraph

# Initialize rich console and traceback handler
install()
console = Console()

# Get file/data paths
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / "Robot_Evolution_TreeGenome")
DATA.mkdir(exist_ok=True, parents=True)


# ============================================================================ #
#                            CONFIGURATION                                     #
# ============================================================================ #

# A seed is optional, but it helps with reproducibility
SEED = 42

# Morphology parameters
NUM_MODULES = 20
MAX_DEPTH = 12  # Maximum tree depth to prevent bloat
POP_SIZE = 15  # Larger population for more diversity
NUM_GENERATIONS = 7  # Longer evolution window

# Controller learning parameters
LEARNING_POP_SIZE = 15  # Larger CMA-ES population
LEARNING_BUDGET = 7  # More learning iterations per morphology
SIMULATION_DURATION = 12  # Slightly longer simulations

# Initialize RNG
RNG = np.random.default_rng(SEED)

# Set config
config = EASettings()
config.is_maximisation = False
config.db_handling = "delete"
config.target_population_size = POP_SIZE
config.output_folder = DATA
config.db_file_name = "database.db"

# Mirror notebook config to module-level config used by EA internals.
engine_config.is_maximisation = config.is_maximisation
engine_config.db_handling = config.db_handling
engine_config.target_population_size = config.target_population_size
engine_config.output_folder = config.output_folder
engine_config.db_file_name = config.db_file_name


# ============================================================================ #
#                        NEURAL NETWORK CONTROLLER                            #
# ============================================================================ #

class Network(nn.Module):
    """Neural network controller with phase inputs for rhythmic gaits."""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 32,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        self.hidden_activation = nn.ELU()
        self.output_activation = nn.Tanh()
        
        # Disable gradients for all parameters
        for param in self.parameters():
            param.requires_grad = False
    
    @torch.inference_mode()
    def forward(self, model, data):
        """Forward pass with phase inputs for temporal awareness."""
        # Get robot state (proprioceptive feedback)
        state = get_state_from_data(data)
        
        # Add phase inputs (sina and cosine provides a circular sense of time)
        # This allows the network to learn rhythmic gaits
        phase_inputs = np.array([
            2 * np.sin(data.time * 2.0 * np.pi),
            2 * np.cos(data.time * 2.0 * np.pi)
        ], dtype=np.float32)
        
        # Concatenate state with phase inputs
        x = torch.Tensor(np.concatenate([state, phase_inputs]))
        
        x = self.hidden_activation(self.fc1(x))
        x = self.hidden_activation(self.fc2(x))
        x = self.output_activation(self.fc3(x)) * (torch.pi / 2)
        
        return x.detach().numpy()


@torch.no_grad()
def fill_parameters(net: nn.Module, vector: np.ndarray) -> None:
    """Fill the parameters of a torch module from a 1-D array."""
    address = 0
    for p in net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = torch.as_tensor(vector[address : address + n], device=d.device, dtype=d.dtype)
        address += n


# ============================================================================ #
#                       MORPHOLOGY UTILITIES                                  #
# ============================================================================ #

def is_connected_tree(genome: TreeGenome) -> bool:
    """Check if genome is a valid connected tree with single root."""
    if len(genome.nodes) == 0:
        return False
    robot_graph = genome.to_networkx()
    # Check for single root (one node with no predecessors)
    roots = [n for n in robot_graph.nodes() if robot_graph.in_degree(n) == 0]
    if len(roots) != 1:
        return False
    # Check connectivity: all nodes reachable from root
    root = roots[0]
    reachable = set()
    stack = [root]
    while stack:
        node = stack.pop()
        reachable.add(node)
        stack.extend(
            succ
            for succ in robot_graph.successors(node)
            if succ not in reachable
        )
    return len(reachable) == robot_graph.number_of_nodes()


def get_module_count(genome: TreeGenome) -> int:
    """Count the number of modules in a morphology."""
    return len(genome.nodes)


def mutate_morphology(genome: TreeGenome) -> TreeGenome:
    """Apply mutations to morphology with multiple GP operators."""
    new = copy.deepcopy(genome)
    
    # Choose mutation type (standard GP mutation operators)
    mutation_type = RNG.choice(
        ["point", "subtree", "shrink", "hoist"], p=[0.4, 0.4, 0.1, 0.1],
    )
    
    if mutation_type == "point":
        # Point mutation: change node type/rotation
        mutate_replace_node(new)
    elif mutation_type == "subtree":
        # Subtree mutation: replace subtree with new random tree
        mutate_subtree_replacement(new, max_modules=NUM_MODULES)
    elif mutation_type == "shrink":
        # Shrink mutation: replace node+subtree with single leaf
        mutate_shrink(new)
    elif mutation_type == "hoist":
        # Hoist mutation: promote child to replace parent
        mutate_hoist(new)
    
    # Additional rotation mutation (20% chance)
    if RNG.random() < 0.2:
        noncore = [nid for nid in new.nodes if nid != IDX_OF_CORE]
        if noncore:
            nid = random.choice(noncore)
            # choose a new rotation allowed for its type
            mtype = ModuleType[new.nodes[nid]["type"]]
            rots = [r.name for r in ALLOWED_ROTATIONS[mtype]]
            if rots:
                new.nodes[nid]["rotation"] = random.choice(rots)
    
    # prune any invalid edges before returning
    _prune_invalid_edges(new)
    with contextlib.suppress(ValueError):
        validate_genome_dict(new.to_dict())
    return new


def crossover_morphologies(parent1: Individual, parent2: Individual) -> TreeGenome:
    """One-point crossover for morphologies. Recovers from invalid results."""
    t1 = parent1.genotype
    t2 = parent2.genotype
    if isinstance(t1, dict):
        t1 = TreeGenome.from_dict(t1)
    if isinstance(t2, dict):
        t2 = TreeGenome.from_dict(t2)
    
    child1, child2 = crossover_subtree(t1, t2)
    
    # Return first valid child, otherwise fallback to asexual copy
    chosen = child1 if RNG.random() < 0.5 else child2
    
    # If disconnected, return copy of a parent instead
    if not is_connected_tree(chosen):
        return TreeGenome.from_dict(random.choice([t1, t2]).to_dict())
    
    return chosen


# ============================================================================ #
#                            EA OPERATIONS                                     #
# ============================================================================ #

def parent_selection(population: Population) -> Population:
    """Select top 50% as parents."""
    population = population.sort(sort="min")
    cutoff = len(population) // 2
    for i, ind in enumerate(population):
        ind.tags["ps"] = i < cutoff
    ps_count = sum(1 for ind in population if ind.tags.get("ps", False))
    console.log(
        f"[cyan]Parent Selection: {ps_count}/{len(population)} marked for reproduction[/cyan]",
    )
    return population


def reproduction(population: Population) -> Population:
    """Create offspring through crossover and mutation."""
    parents = [ind for ind in population if ind.tags.get("ps", False)]
    if not parents:
        console.log(
            "[yellow]Warning: No ps-tagged individuals, using entire population[/yellow]",
        )
        parents = population
    
    new_offspring: list[Individual] = []
    target_pool = config.target_population_size * 2
    while len(population) + len(new_offspring) < target_pool:
        use_sexual = len(parents) >= 2 and RNG.random() < 0.5
        if use_sexual:
            p1, p2 = random.sample(parents, 2)
            c_morph = crossover_morphologies(p1, p2)
        else:
            parent = random.choice(parents)
            c_morph = TreeGenome.from_dict(parent.genotype)
        
        # mutate morphology
        c_morph = mutate_morphology(c_morph)
        
        # body validation loop: ensure valid morphology
        valid_child = False
        attempts = 0
        while not valid_child and attempts < 20:
            has_modules = get_module_count(c_morph) > 0
            valid_depth = validate_tree_depth(c_morph, MAX_DEPTH)
            if has_modules and valid_depth:
                valid_child = True
            else:
                c_morph = mutate_morphology(c_morph)
            attempts += 1
        
        ind = Individual()
        ind.genotype = c_morph.to_dict()
        ind.tags["valid"] = True
        ind.requires_eval = True
        new_offspring.append(ind)
    
    population.extend(new_offspring)
    return population


def evaluate(population: Population) -> Population:
    """Evaluate population by learning optimal neural network controllers."""
    to_eval = [
        ind
        for ind in population
        if ind.alive and ind.tags.get("valid") and ind.requires_eval
    ]
    if not to_eval:
        return population
    
    for ind in track(to_eval, description="Evaluating..."):
        genome = TreeGenome.from_dict(ind.genotype)
        
        # Get morphology from genome
        try:
            robot_graph = genome.to_networkx()
            if robot_graph.number_of_nodes() == 0:
                ind.fitness = float("inf")
                ind.requires_eval = False
                continue
            
            spec = construct_mjspec_from_graph(robot_graph).spec
            
            # Set up world and robot
            mj.set_mjcb_control(None)
            world = SimpleFlatWorld()
            world.spawn(spec, position=(0.0, 0.0, 0.1), correct_collision_with_floor=True)
            
            model = world.spec.compile()
            data = mj.MjData(model)
            
            # Create neural network controller
            net = Network(
                input_size=len(get_state_from_data(data)) + 2,  # +2 for phase inputs
                output_size=model.nu,
                hidden_size=32,
            )
            
            # Get number of network parameters
            num_vars = sum(p.numel() for p in net.parameters())
            
            # Initialize CMA-ES optimizer
            param = ng.p.Array(shape=(num_vars,))
            optimizer = ng.optimizers.registry["CMA"](
                parametrization=param,
                budget=(LEARNING_POP_SIZE * LEARNING_BUDGET),
            )
            
            # Set up tracker and controller
            tracker = Tracker(
                name_to_bind="core",
                observable_attributes=["xpos"],
                quiet=True,
            )
            tracker.setup(world.spec, data)
            
            controller = Controller(
                controller_callback_function=net.forward,
                tracker=tracker,
            )
            
            # CMA-ES learning loop
            min_fit = np.inf
            best_weights = None
            for _ in range(LEARNING_BUDGET):
                candidates = [optimizer.ask() for _ in range(LEARNING_POP_SIZE)]
                
                for candidate in candidates:
                    weights = candidate.value
                    fill_parameters(net, weights)
                    
                    # Reset simulation
                    mj.mj_resetData(model, data)
                    mj.set_mjcb_control(controller.set_control)
                    
                    # Run simulation
                    simple_runner(model, data, duration=SIMULATION_DURATION)
                    
                    # Calculate fitness (distance to target)
                    xc, yc, zc = data.qpos[0:3].copy()
                    xt, yt, zt = (2.0, 0.0, 0.1)
                    fitness = np.sqrt((xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2)
                    
                    optimizer.tell(candidate, fitness)
                    if fitness < min_fit:
                        min_fit = fitness
                        best_weights = np.array(weights, copy=True)
            
            ind.fitness = min_fit
            # Store best weights as list (numpy array can't be JSON serialized)
            if best_weights is not None:
                ind.tags["best_weights"] = best_weights.tolist()
            else:
                ind.tags["best_weights"] = None
            
        except Exception as e:
            console.log(f"[yellow]Evaluation error: {e}[/yellow]")
            ind.fitness = float("inf")
        
        ind.requires_eval = False
    
    return population


def survivor_selection(population: Population) -> Population:
    """Keep top 50% as survivors."""
    population = population.sort(sort="min", attribute="fitness_")
    survivors = population[: config.target_population_size]
    for ind in population:
        if ind not in survivors:
            ind.alive = False
    
    # Print statistics
    avg_fitness = np.mean([
        ind.fitness_
        for ind in survivors
        if ind.fitness_ is not None and ind.fitness_ != float("inf")
    ])
    min_fitness = min(
        ind.fitness_
        for ind in survivors
        if ind.fitness_ is not None and ind.fitness_ != float("inf")
    )
    max_fitness = max(
        ind.fitness_
        for ind in survivors
        if ind.fitness_ is not None and ind.fitness_ != float("inf")
    )
    
    console.log(
        f"[green]Survivor Selection:[/green] "
        f"Avg={avg_fitness:.4f}, Min={min_fitness:.4f}, Max={max_fitness:.4f}",
    )
    return population


# ============================================================================ #
#                         MAIN EVOLUTION LOOP                                 #
# ============================================================================ #

def evolve() -> EA:
    """Entry point for the evolutionary algorithm.

    Initializes the population, evaluates it, defines the evolutionary 
    steps (parent selection, reproduction, evaluation, survivor selection),
    and runs the EA.

    Returns
    -------
    EA
        The completed EA object containing the run history and solutions.
    """
    console.log("Initializing population...")
    
    # Create initial population with random tree genomes
    population = []
    for _ in range(POP_SIZE):
        ind = Individual()
        ind.genotype = random_tree(NUM_MODULES).to_dict()
        ind.tags["valid"] = True
        ind.requires_eval = True
        population.append(ind)
    population = Population(population)
    
    # Initial evaluation
    population = evaluate(population)
    
    # Define evolutionary operators
    ops = [
        EAOperation(parent_selection),
        EAOperation(reproduction),
        EAOperation(evaluate),
        EAOperation(survivor_selection),
    ]
    
    # Initialize and run EA
    ea = EA(
        population,
        operations=ops,
        num_steps=NUM_GENERATIONS,
        db_file_path=config.db_file_path,
        db_handling=config.db_handling,
    )
    ea.run()
    
    return ea


def main() -> EA:
    """Main entry point."""
    console.rule(
        "[bold purple]Starting Robot Evolution with TreeGenome + Neural Controllers[/bold purple]",
    )
    console.log(
        f"Population: {POP_SIZE}, Generations: {NUM_GENERATIONS}, "
        f"Modules: {NUM_MODULES}, Learning Budget: {LEARNING_BUDGET}",
    )
    
    start_time = time.time()
    ea = evolve()
    elapsed = time.time() - start_time
    
    # Get and visualize best solution
    best_individual = ea.get_solution("best", only_alive=False)
    if best_individual:
        console.rule("[bold green]Final Best Result[/bold green]")
        console.log(f"Best Fitness: {best_individual.fitness_:.4f}")
        console.log(f"Runtime: {elapsed/60:.2f} minutes")
        
        # Visualize best morphology
        try:
            genome = TreeGenome.from_dict(best_individual.genotype)
            robot_graph = genome.to_networkx()
            spec = construct_mjspec_from_graph(robot_graph).spec
            
            mj.set_mjcb_control(None)
            world = SimpleFlatWorld()
            world.spawn(spec, position=(0.0, 0.0, 0.1), correct_collision_with_floor=True)
            model = world.spec.compile()
            data = mj.MjData(model)
            
            # Create and initialize neural network with learned weights
            net = Network(
                input_size=len(get_state_from_data(data)) + 2,
                output_size=model.nu,
                hidden_size=32,
            )
            
            # Load best learned weights if available
            best_weights = best_individual.tags.get("best_weights")
            if best_weights is not None:
                # Convert list back to numpy array
                best_weights = np.array(best_weights, dtype=np.float64)
                fill_parameters(net, best_weights)
                console.log(f"[cyan]Loaded learned controller weights[/cyan]")
            else:
                console.log(f"[yellow]No learned weights available, using random initialization[/yellow]")
            
            # Set up controller
            tracker = Tracker(
                name_to_bind="core",
                observable_attributes=["xpos"],
                quiet=True,
            )
            tracker.setup(world.spec, data)
            
            controller = Controller(
                controller_callback_function=net.forward,
                tracker=tracker,
            )
            mj.set_mjcb_control(controller.set_control)
            
            console.log(f"[cyan]Launching MuJoCo viewer for best morphology[/cyan]")
            viewer.launch(model=model, data=data)
        except Exception as e:
            console.log(f"[yellow]Could not visualize: {e}[/yellow]")


if __name__ == "__main__":
    main()
