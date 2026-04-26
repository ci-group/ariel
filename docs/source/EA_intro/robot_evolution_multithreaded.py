"""
Robot Evolution Example using TreeGenome Morphologies with Neural Network Controllers

This script demonstrates joint-evolution of robot morphologies and controllers:
- **Morphologies**: Represented using TreeGenome (fast genetic programming approach)
- **Controllers**: Neural networks with phase inputs trained via CMA-ES learning

The example evolves morphologically diverse robots to move towards a target position.
"""

import contextlib
import copy
import os
import random
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import mujoco as mj
import nevergrad as ng
import numpy as np
import torch
from mujoco import viewer
from rich.console import Console
from rich.traceback import install
from torch import nn

# Ariel Imports
from ariel.body_phenotypes.robogen_lite.config import ALLOWED_ROTATIONS, IDX_OF_CORE, ModuleType
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.ec import EA, EAOperation, EASettings, Individual, Population
from ariel.ec import config as engine_config
from ariel.ec.genotypes.tree.operators import (
    _prune_invalid_edges, crossover_subtree, mutate_hoist,
    mutate_replace_node, mutate_shrink, mutate_subtree_replacement,
    random_tree, validate_tree_depth,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict
from ariel.simulation.controllers.utils.data_get import get_state_from_data
from ariel.simulation.environments import SimpleFlatWorld

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

SEED = 42
NUM_MODULES = 20
MAX_DEPTH = 12
POP_SIZE = 16 
NUM_GENERATIONS = 5

LEARNING_POP_SIZE = 6  # Slightly bumped for better CMA-ES distribution
LEARNING_BUDGET = 8 
SIMULATION_DURATION = 5
# EVAL_WORKERS = max(1, min(8, os.cpu_count() or 1))
EVAL_WORKERS = 8

RNG = np.random.default_rng(SEED)
_LOG_LOCK = threading.Lock()

config = EASettings()
config.is_maximisation = False
config.db_handling = "delete"
config.target_population_size = POP_SIZE
config.output_folder = DATA
config.db_file_name = "database.db"

engine_config.is_maximisation = config.is_maximisation
engine_config.db_handling = config.db_handling
engine_config.target_population_size = config.target_population_size
engine_config.output_folder = config.output_folder
engine_config.db_file_name = config.db_file_name

# ============================================================================ #
#                        NEURAL NETWORK CONTROLLER                             #
# ============================================================================ #

class Network(nn.Module):
    """Streamlined Neural network controller for faster CMA-ES convergence."""
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 16) -> None:
        super().__init__()
        # Reduced to a single hidden layer. Smaller parameter space = faster learning!
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        self.hidden_activation = nn.ELU()
        self.output_activation = nn.Tanh()
        
        for param in self.parameters():
            param.requires_grad = False
    
    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> np.ndarray:
        x = self.hidden_activation(self.fc1(x))
        x = self.output_activation(self.fc2(x)) * (torch.pi / 2)
        return x.detach().numpy()

@torch.no_grad()
def fill_parameters(net: nn.Module, vector: np.ndarray) -> None:
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
    if len(genome.nodes) == 0:
        return False
    robot_graph = genome.to_networkx()
    roots = [n for n in robot_graph.nodes() if robot_graph.in_degree(n) == 0]
    if len(roots) != 1:
        return False
    root = roots[0]
    reachable = set()
    stack = [root]
    while stack:
        node = stack.pop()
        reachable.add(node)
        stack.extend(succ for succ in robot_graph.successors(node) if succ not in reachable)
    return len(reachable) == robot_graph.number_of_nodes()

def get_module_count(genome: TreeGenome) -> int:
    return len(genome.nodes)

def mutate_morphology(genome: TreeGenome) -> TreeGenome:
    new = copy.deepcopy(genome)
    mutation_type = RNG.choice(["point", "subtree", "shrink", "hoist"], p=[0.4, 0.4, 0.1, 0.1])
    
    if mutation_type == "point": mutate_replace_node(new)
    elif mutation_type == "subtree": mutate_subtree_replacement(new, max_modules=NUM_MODULES)
    elif mutation_type == "shrink": mutate_shrink(new)
    elif mutation_type == "hoist": mutate_hoist(new)
    
    if RNG.random() < 0.2:
        noncore = [nid for nid in new.nodes if nid != IDX_OF_CORE]
        if noncore:
            nid = random.choice(noncore)
            mtype = ModuleType[new.nodes[nid]["type"]]
            rots = [r.name for r in ALLOWED_ROTATIONS[mtype]]
            if rots:
                new.nodes[nid]["rotation"] = random.choice(rots)
    
    _prune_invalid_edges(new)
    with contextlib.suppress(ValueError):
        validate_genome_dict(new.to_dict())
    return new

def crossover_morphologies(parent1: Individual, parent2: Individual) -> TreeGenome:
    t1 = TreeGenome.from_dict(parent1.genotype) if isinstance(parent1.genotype, dict) else parent1.genotype
    t2 = TreeGenome.from_dict(parent2.genotype) if isinstance(parent2.genotype, dict) else parent2.genotype
    
    child1, child2 = crossover_subtree(t1, t2)
    chosen = child1 if RNG.random() < 0.5 else child2
    
    if not is_connected_tree(chosen):
        return TreeGenome.from_dict(random.choice([t1, t2]).to_dict())
    return chosen

# ============================================================================ #
#                            EA OPERATIONS                                     #
# ============================================================================ #

def parent_selection(population: Population) -> Population:
    population = population.sort(sort="min")
    cutoff = len(population) // 2
    for i, ind in enumerate(population):
        ind.tags["ps"] = i < cutoff
    return population

def reproduction(population: Population) -> Population:
    parents = [ind for ind in population if ind.tags.get("ps", False)]
    if not parents: parents = population
    
    new_offspring: list[Individual] = []
    target_pool = config.target_population_size * 2
    
    while len(population) + len(new_offspring) < target_pool:
        if len(parents) >= 2 and RNG.random() < 0.5:
            p1, p2 = random.sample(parents, 2)
            c_morph = crossover_morphologies(p1, p2)
        else:
            c_morph = TreeGenome.from_dict(random.choice(parents).genotype)
        
        c_morph = mutate_morphology(c_morph)
        
        valid_child = False
        attempts = 0
        while not valid_child and attempts < 20:
            if get_module_count(c_morph) > 0 and validate_tree_depth(c_morph, MAX_DEPTH):
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

def evaluate_single_individual(ind: Individual) -> tuple[float, list | None]:
    """Evaluates a single robot morphology by training its brain using CMA-ES."""
    genome = TreeGenome.from_dict(ind.genotype)
    
    try:
        robot_graph = genome.to_networkx()
        if robot_graph.number_of_nodes() == 0:
            return float("inf"), None
        
        spec = construct_mjspec_from_graph(robot_graph).spec
        world = SimpleFlatWorld()
        world.spawn(spec, position=(0.0, 0.0, 0.1), correct_collision_with_floor=True)
        model = world.spec.compile()
        data = mj.MjData(model)
        
        # Determine sizes and initialize small network
        state_size = len(get_state_from_data(data))
        net = Network(input_size=state_size + 2, output_size=model.nu, hidden_size=16)
        
        # Setup modern ParametrizedCMA
        num_vars = sum(p.numel() for p in net.parameters())
        param = ng.p.Array(init=np.zeros(num_vars))
        param.set_mutation(sigma=0.1) # Smaller steps for stable NN weight tuning
        
        cma_config = ng.optimizers.ParametrizedCMA(popsize=LEARNING_POP_SIZE)
        optimizer = cma_config(parametrization=param, budget=(LEARNING_POP_SIZE * LEARNING_BUDGET))
        
        min_fit = float("inf")
        best_weights = None

        with _LOG_LOCK:
            console.log("[blue]Worker started one individual evaluation[/blue]")
        
        # Standard batched CMA-ES Loop
        for learning_step in range(LEARNING_BUDGET):
            candidates = [optimizer.ask() for _ in range(LEARNING_POP_SIZE)]
            fitnesses = []
            
            for candidate in candidates:
                fill_parameters(net, candidate.value)
                mj.mj_resetData(model, data)
                
                # Raw, highly optimized simulation loop
                while data.time < SIMULATION_DURATION:
                    state = get_state_from_data(data)
                    phase = np.array([np.sin(data.time * 2.0 * np.pi), np.cos(data.time * 2.0 * np.pi)], dtype=np.float32)
                    x = torch.Tensor(np.concatenate([state, phase]))
                    
                    data.ctrl[:] = net.forward(x)
                    mj.mj_step(model, data)
                
                # Planar 2D Fitness (Distance to 2.0, 0.0)
                xc, yc = data.qpos[0:2]
                fitness = float(np.sqrt((2.0 - xc)**2 + (0.0 - yc)**2))
                fitnesses.append(fitness)
                
                if fitness < min_fit:
                    min_fit = fitness
                    best_weights = np.array(candidate.value, copy=True)
            
            # Batch tell
            for cand, fit in zip(candidates, fitnesses):
                optimizer.tell(cand, fit)

            if (
                learning_step == 0
                or (learning_step + 1) % max(1, LEARNING_BUDGET // 4) == 0
                or learning_step + 1 == LEARNING_BUDGET
            ):
                with _LOG_LOCK:
                    console.log(
                        f"[blue]Individual progress:[/blue] "
                        f"{learning_step + 1}/{LEARNING_BUDGET} "
                        f"learning batches | best fit: {min_fit:.4f}"
                    )
                
        return min_fit, best_weights.tolist() if best_weights is not None else None
        
    except Exception as e:
        return float("inf"), None

def evaluate(population: Population) -> Population:
    to_eval = [ind for ind in population if ind.alive and ind.tags.get("valid") and ind.requires_eval]
    if not to_eval: return population
    
    console.log(
        f"[cyan]Dispatching {len(to_eval)} evaluations to ThreadPool "
        f"(workers={EVAL_WORKERS})...[/cyan]"
    )

    results: list[tuple[float, list | None] | None] = [None] * len(to_eval)
    completed = 0
    heartbeat_seconds = 10.0
    start_time = time.monotonic()

    with ThreadPoolExecutor(max_workers=EVAL_WORKERS) as executor:
        future_to_idx = {
            executor.submit(evaluate_single_individual, ind): idx
            for idx, ind in enumerate(to_eval)
        }

        pending = set(future_to_idx)
        while pending:
            done, pending = wait(
                pending,
                timeout=heartbeat_seconds,
                return_when=FIRST_COMPLETED,
            )

            if not done:
                elapsed = time.monotonic() - start_time
                console.log(
                    f"[cyan]Evaluation heartbeat:[/cyan] {completed}/{len(to_eval)} "
                    f"completed | elapsed {elapsed:.0f}s"
                )
                continue

            for future in done:
                idx = future_to_idx[future]
                results[idx] = future.result()
                completed += 1

                if completed == 1 or completed % max(1, len(to_eval) // 4) == 0 or completed == len(to_eval):
                    done_fits = [
                        res[0]
                        for res in results
                        if res is not None and np.isfinite(res[0])
                    ]
                    best_so_far = min(done_fits) if done_fits else float("inf")
                    elapsed = time.monotonic() - start_time
                    console.log(
                        f"[cyan]Evaluation progress:[/cyan] {completed}/{len(to_eval)} "
                        f"| best so far: {best_so_far:.4f} | elapsed {elapsed:.0f}s"
                    )

    for ind, result in zip(to_eval, results):
        fitness, best_weights = result if result is not None else (float("inf"), None)
        ind.fitness = fitness
        ind.tags["best_weights"] = best_weights
        ind.requires_eval = False
    
    return population

def survivor_selection(population: Population) -> Population:
    population = population.sort(sort="min", attribute="fitness_")
    survivors = population[: config.target_population_size]
    for ind in population:
        if ind not in survivors:
            ind.alive = False
            
    valid_fitnesses = [ind.fitness_ for ind in survivors if ind.fitness_ is not None and ind.fitness_ != float("inf")]
    if valid_fitnesses:
        console.log(f"[green]Survivor Selection:[/green] Avg={np.mean(valid_fitnesses):.4f}, Min={min(valid_fitnesses):.4f}, Max={max(valid_fitnesses):.4f}")
    return population

# ============================================================================ #
#                         MAIN EVOLUTION LOOP                                 #
# ============================================================================ #

def evolve() -> EA:
    console.log("Initializing population...")
    population = []
    for _ in range(POP_SIZE):
        ind = Individual()
        ind.genotype = random_tree(NUM_MODULES).to_dict()
        ind.tags["valid"] = True
        ind.requires_eval = True
        population.append(ind)
    
    population = Population(population)
    population = evaluate(population)
    
    ops = [
        EAOperation(parent_selection),
        EAOperation(reproduction),
        EAOperation(evaluate),
        EAOperation(survivor_selection),
    ]
    
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
    console.rule("[bold purple]Starting Fast Robot Evolution with TreeGenome[/bold purple]")
    
    start_time = time.time()
    ea = evolve()
    elapsed = time.time() - start_time
    
    best_individual = ea.get_solution("best", only_alive=False)
    if best_individual:
        console.rule("[bold green]Final Best Result[/bold green]")
        console.log(f"Best Fitness: {best_individual.fitness_:.4f}")
        console.log(f"Runtime: {elapsed/60:.2f} minutes")
        
        try:
            genome = TreeGenome.from_dict(best_individual.genotype)
            robot_graph = genome.to_networkx()
            spec = construct_mjspec_from_graph(robot_graph).spec
            
            mj.set_mjcb_control(None)
            world = SimpleFlatWorld()
            world.spawn(spec, position=(0.0, 0.0, 0.1), correct_collision_with_floor=True)
            model = world.spec.compile()
            data = mj.MjData(model)
            
            net = Network(input_size=len(get_state_from_data(data)) + 2, output_size=model.nu, hidden_size=16)
            
            if best_individual.tags.get("best_weights") is not None:
                fill_parameters(net, np.array(best_individual.tags.get("best_weights"), dtype=np.float64))
                console.log("[cyan]Loaded learned controller weights[/cyan]")
            
            console.log("[cyan]Launching MuJoCo viewer for best morphology...[/cyan]")
            with viewer.launch_passive(model, data) as v:
                while v.is_running():
                    state = get_state_from_data(data)
                    phase = np.array([np.sin(data.time * 2.0 * np.pi), np.cos(data.time * 2.0 * np.pi)], dtype=np.float32)
                    x = torch.Tensor(np.concatenate([state, phase]))
                    
                    data.ctrl[:] = net.forward(x)
                    mj.mj_step(model, data)
                    
                    v.sync()
                    time.sleep(model.opt.timestep)

        except Exception as e:
            console.log(f"[yellow]Could not visualize: {e}[/yellow]")

if __name__ == "__main__":
    main()