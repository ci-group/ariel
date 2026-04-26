"""
Runner: Load and simulate a saved morphology + brain.

Usage:
    uv run runner_best_individual.py \
      --morphology path/to/best_morphology.json \
      --brain path/to/best_brain.npy \
      --duration 10.0
"""

# Standard library
import argparse
import sys
import time
from pathlib import Path

# Third-party
import mujoco
import mujoco.viewer
import numpy as np
import torch
from rich.console import Console
from rich.traceback import install
from torch import nn

# ARIEL imports
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.simulation.controllers.controller import Controller, Tracker
from ariel.simulation.controllers.utils.data_get import get_state_from_data
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.runners import thread_safe_runner

install()
console = Console()


# ============================================================================ #
#                                   CONFIG                                     #
# ============================================================================ #

parser = argparse.ArgumentParser(description="Load and run a saved morphology + brain")

parser.add_argument(
    "--morphology",
    type=str,
    required=True,
    help="Path to best_morphology.json",
)
parser.add_argument(
    "--brain",
    type=str,
    required=True,
    help="Path to best_brain.npy",
)
parser.add_argument(
    "--duration",
    type=float,
    default=10.0,
    help="Simulation duration in seconds",
)
parser.add_argument(
    "--spawn-pos-x",
    type=float,
    default=-0.8,
    help="Spawn position X",
)
parser.add_argument(
    "--spawn-pos-y",
    type=float,
    default=0.0,
    help="Spawn position Y",
)
parser.add_argument(
    "--spawn-pos-z",
    type=float,
    default=0.1,
    help="Spawn position Z",
)

args = parser.parse_args()

MORPHOLOGY_PATH = Path(args.morphology)
BRAIN_PATH = Path(args.brain)
DURATION = args.duration
SPAWN_POSITION = (args.spawn_pos_x, args.spawn_pos_y, args.spawn_pos_z)


# ============================================================================ #
#                                  NETWORK                                     #
# ============================================================================ #


class Network(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.hidden_activation = nn.ELU()
        self.output_activation = nn.Tanh()

        for param in self.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def forward(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        robot_state = get_state_from_data(data)
        phase_inputs = np.array(
            [
                2.0 * np.sin(data.time * 2.0 * np.pi),
                2.0 * np.cos(data.time * 2.0 * np.pi),
            ],
            dtype=np.float32,
        )
        state = torch.tensor(
            np.concatenate([robot_state, phase_inputs]).astype(np.float32),
            dtype=torch.float32,
        )
        x = self.hidden_activation(self.fc1(state))
        x = self.hidden_activation(self.fc2(x))
        x = self.output_activation(self.fc_out(x)) * (torch.pi / 2)
        return x.detach().numpy()


@torch.no_grad()
def fill_parameters(net: nn.Module, vector: np.ndarray | list[float]) -> None:
    address = 0
    for p in net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = torch.as_tensor(vector[address : address + n], device=d.device)
        address += n


# ============================================================================ #
#                                    MAIN                                      #
# ============================================================================ #


def main() -> None:
    # Validate files exist
    if not MORPHOLOGY_PATH.exists():
        console.log(f"[red]Morphology file not found: {MORPHOLOGY_PATH}[/red]")
        sys.exit(1)

    if not BRAIN_PATH.exists():
        console.log(f"[red]Brain file not found: {BRAIN_PATH}[/red]")
        sys.exit(1)

    # Load morphology
    try:
        import json
        morph_dict = json.loads(MORPHOLOGY_PATH.read_text())
        genome = TreeGenome.from_dict(morph_dict)
        graph = genome.to_networkx()
        if graph.number_of_nodes() == 0:
            console.log("[red]Morphology is empty.[/red]")
            sys.exit(1)
        console.log(f"[green]Loaded morphology from {MORPHOLOGY_PATH}[/green]")
    except Exception as exc:
        console.log(f"[red]Failed to load morphology: {exc}[/red]")
        sys.exit(1)

    # Build MuJoCo model
    try:
        spec = construct_mjspec_from_graph(graph).spec
        world = SimpleFlatWorld()
        world.spawn(spec, position=SPAWN_POSITION, correct_collision_with_floor=True)
        model = world.spec.compile()
        data = mujoco.MjData(model)
        console.log(f"[green]Built MuJoCo model with {model.nu} actuators[/green]")
    except Exception as exc:
        console.log(f"[red]Failed to build MuJoCo model: {exc}[/red]")
        sys.exit(1)

    # Load brain
    try:
        brain_vec = np.load(BRAIN_PATH)
        console.log(f"[green]Loaded brain from {BRAIN_PATH} ({len(brain_vec)} parameters)[/green]")
    except Exception as exc:
        console.log(f"[red]Failed to load brain: {exc}[/red]")
        sys.exit(1)

    # Create network and fill parameters
    try:
        input_size = len(get_state_from_data(data)) + 2
        net = Network(input_size=input_size, output_size=model.nu, hidden_size=16)
        fill_parameters(net, brain_vec)
        console.log(f"[green]Initialized network: input={input_size}, output={model.nu}[/green]")
    except Exception as exc:
        console.log(f"[red]Failed to initialize network: {exc}[/red]")
        sys.exit(1)

    # Setup tracker and controller
    try:
        tracker = Tracker(name_to_bind="core", observable_attributes=["xpos"], quiet=True)
        tracker.setup(world.spec, data)
        controller = Controller(controller_callback_function=net.forward, tracker=tracker)
        console.log("[green]Controller ready[/green]")
    except Exception as exc:
        console.log(f"[red]Failed to setup controller: {exc}[/red]")
        sys.exit(1)

    # Run simulation with viewer
    console.rule("[bold cyan]Starting Simulation[/bold cyan]")
    console.log(f"Duration: {DURATION}s | Spawn: {SPAWN_POSITION}")
    console.log("[yellow]Close the viewer window to exit.[/yellow]")

    mujoco.mj_resetData(model, data)

    try:
        if sys.platform == "darwin" or not hasattr(mujoco.viewer, "launch_passive"):
            console.log("[yellow]Using active MuJoCo viewer fallback.[/yellow]")
            mujoco.set_mjcb_control(controller.set_control)
            mujoco.viewer.launch(model=model, data=data)
        else:
            with mujoco.viewer.launch_passive(model, data) as v:
                sim_start = time.time()
                while v.is_running() and (time.time() - sim_start) < DURATION:
                    step_start = time.time()
                    controller.set_control(model, data)
                    mujoco.mj_step(model, data)
                    v.sync()
                    remaining = model.opt.timestep - (time.time() - step_start)
                    if remaining > 0:
                        time.sleep(remaining)
    finally:
        mujoco.set_mjcb_control(None)

    console.rule("[bold green]Simulation Complete[/bold green]")


if __name__ == "__main__":
    main()
