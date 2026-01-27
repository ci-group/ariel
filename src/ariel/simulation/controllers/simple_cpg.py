"""
Simplified Central Pattern Generator (CPG) using coupled Hopf oscillators.

This is a faster, simpler alternative to NA-CPG that should learn more efficiently.
Uses classic coupled oscillator dynamics with an identical API.

References
----------
    [1] Hopf oscillator formulation for locomotion control
    [2] Phase-coupled oscillators for motor control
"""

# Standard library
from pathlib import Path

# Third-party libraries
import numpy as np
import torch
from rich.console import Console
from rich.traceback import install
from torch import nn

# Global constants
E = 1e-9

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(exist_ok=True)

# --- TERMINAL OUTPUT SETUP ---
install(show_locals=False)
console = Console()
torch.set_printoptions(precision=4)


def create_fully_connected_adjacency(num_nodes: int) -> dict[int, list[int]]:
    """
    Create a fully connected adjacency dictionary for the CPG network.
    
    Parameters
    ----------
    num_nodes : int
        Number of nodes in the CPG network.
    
    Returns
    -------
    dict[int, list[int]]
        Adjacency dictionary where each key is a node index and the value is a list
        of indices of connected nodes.
    """
    adjacency_dict = {}
    for i in range(num_nodes):
        adjacency_dict[i] = [j for j in range(num_nodes) if j != i]
    return adjacency_dict


class SimpleCPG(nn.Module):
    """
    Simplified CPG using coupled Hopf oscillators.
    
    Dynamics:
        dx_i/dt = (mu - r_i^2) * x_i - w_i * y_i + sum_j(coupling_ij * (x_j - x_i))
        dy_i/dt = (mu - r_i^2) * y_i + w_i * x_i + sum_j(coupling_ij * (y_j - y_i))
        
    Where r_i = sqrt(x_i^2 + y_i^2)
    """

    def __init__(
        self,
        adjacency_dict: dict[int, list[int]],
        mu: float = 1.0,
        dt: float = 0.01,
        hard_bounds: tuple[float, float] | None = (-torch.pi / 2, torch.pi / 2),
        *,
        angle_tracking: bool = False,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the SimpleCPG module.

        Parameters
        ----------
        adjacency_dict : dict[int, list[int]]
            Dictionary defining the connectivity of the CPG network.
        mu : float, optional
            Stability parameter (controls oscillation amplitude), by default 1.0
        dt : float, optional
            Time step for updates, by default 0.01
        hard_bounds : tuple[float, float] | None, optional
            Output angle bounds, by default (-π/2, π/2)
        angle_tracking : bool, optional
            Whether to store angle history, by default False
        seed : int | None, optional
            Random seed for reproducibility, by default None
        """
        super().__init__()

        self.adjacency_dict = adjacency_dict
        self.n = len(adjacency_dict)
        self.mu = mu
        self.dt = dt
        self.angle_tracking = angle_tracking
        self.hard_bounds = hard_bounds
        self.clamping_error = 0.0

        if seed is not None:
            torch.manual_seed(seed)

        # ================================================================== #
        # Learnable parameters - keep same naming as NA-CPG for compatibility
        # ================================================================== #
        
        # Phase offset for each oscillator
        self.phase = nn.Parameter(
            (torch.rand(self.n) * 2 - 1) * torch.pi,
            requires_grad=False,
        )
        
        # Frequency for each oscillator
        self.w = nn.Parameter(
            (torch.rand(self.n) * 2 - 1) * 4.0 + 2.0,  # Range [0, 4] Hz roughly
            requires_grad=False,
        )
        
        # Amplitude for each oscillator
        self.amplitudes = nn.Parameter(
            (torch.rand(self.n) * 2 - 1) * 2.0 + 1.0,  # Range [0, 2]
            requires_grad=False,
        )
        
        # Coupling strengths (replaces 'ha' - coupling weight/asymmetry)
        self.ha = nn.Parameter(
            torch.randn(self.n) * 0.5,
            requires_grad=False,
        )
        
        # Bias/offset parameter (minor role)
        self.b = nn.Parameter(
            torch.randn(self.n) * 0.1,
            requires_grad=False,
        )

        self.parameter_groups = {
            "phase": self.phase,
            "w": self.w,
            "amplitudes": self.amplitudes,
            "ha": self.ha,
            "b": self.b,
        }
        self.num_of_parameters = sum(p.numel() for p in self.parameters())
        self.num_of_parameter_groups = len(self.parameter_groups)

        # ================================================================== #
        # Internal state buffers
        # ================================================================== #
        self.register_buffer("x", torch.randn(self.n) * 0.1)
        self.register_buffer("y", torch.randn(self.n) * 0.1)
        self.register_buffer("angles", torch.zeros(self.n))
        
        self.angle_history = []
        self.initial_state = {
            "x": self.x.clone(),
            "y": self.y.clone(),
            "angles": self.angles.clone(),
        }

    def param_type_converter(
        self,
        params: list[float] | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Convert input parameters to torch.Tensor if needed."""
        if isinstance(params, list):
            params = torch.tensor(params, dtype=torch.float32)
        elif isinstance(params, np.ndarray):
            params = torch.from_numpy(params).float()
        return params

    def set_flat_params(self, params: torch.Tensor) -> None:
        """Set all learnable parameters from a flat tensor."""
        safe_params = self.param_type_converter(params)

        if safe_params.numel() != self.num_of_parameters:
            msg = f"Parameter vector has incorrect size. Expected {self.num_of_parameters}, got {safe_params.numel()}."
            raise ValueError(msg)

        pointer = 0
        for param in self.parameter_groups.values():
            num_param = param.numel()
            param.data = safe_params[pointer : pointer + num_param].view_as(param)
            pointer += num_param

    def set_param_with_dict(self, params: dict[str, torch.Tensor]) -> None:
        """Set parameters using a dictionary."""
        for key, value in params.items():
            safe_value = self.param_type_converter(value)
            self.set_params_by_group(key, safe_value)

    def set_params_by_group(
        self,
        group_name: str,
        params: torch.Tensor,
    ) -> None:
        """Set parameters for a specific group."""
        safe_params = self.param_type_converter(params)

        if group_name not in self.parameter_groups:
            raise ValueError(f"Parameter group '{group_name}' does not exist.")

        param = self.parameter_groups[group_name]
        if safe_params.numel() != param.numel():
            raise ValueError(f"Parameter vector has incorrect size for group '{group_name}'.")
        param.data = safe_params.view_as(param)

    def get_flat_params(self) -> torch.Tensor:
        """Get all learnable parameters as a flat tensor."""
        return torch.cat([p.flatten() for p in self.parameter_groups.values()])

    def reset(self) -> None:
        """Reset the internal states to their initial values."""
        self.x.data = self.initial_state["x"].clone()
        self.y.data = self.initial_state["y"].clone()
        self.angles.data = self.initial_state["angles"].clone()
        self.angle_history = []

    def forward(self, time: float | None = None) -> torch.Tensor:
        """
        Perform a forward pass to update CPG states and compute output angles.
        
        Parameters
        ----------
        time : float | None, optional
            Current simulation time. If equal to zero, resets the CPG.
        
        Returns
        -------
        torch.Tensor
            Output angles for each CPG node.
        """
        # Reset if time is zero
        if time is not None and torch.isclose(
            torch.tensor(time),
            torch.tensor(0.0),
        ):
            self.reset()

        with torch.inference_mode():
            # Compute radius for each oscillator
            r = torch.sqrt(self.x**2 + self.y**2 + E)
            
            # Hopf oscillator dynamics with coupling
            dx = torch.zeros_like(self.x)
            dy = torch.zeros_like(self.y)
            
            for i in range(self.n):
                # Local Hopf dynamics: (mu - r^2) * state + w * perpendicular
                dx[i] = (self.mu - r[i]**2) * self.x[i] - self.w[i] * self.y[i]
                dy[i] = (self.mu - r[i]**2) * self.y[i] + self.w[i] * self.x[i]
                
                # Coupling from connected oscillators
                coupling_strength = self.ha[i]
                for j in self.adjacency_dict[i]:
                    # Phase difference based coupling
                    phase_diff = torch.atan2(self.y[j], self.x[j] + E) - torch.atan2(self.y[i], self.x[i] + E)
                    dx[i] += coupling_strength * torch.sin(phase_diff) * (self.x[j] - self.x[i])
                    dy[i] += coupling_strength * torch.sin(phase_diff) * (self.y[j] - self.y[i])
            
            # Update states
            self.x = self.x + dx * self.dt
            self.y = self.y + dy * self.dt
            
            # Compute output angles: amplitude * y component + phase offset
            angles = self.amplitudes * self.y + self.phase + self.b
            
            # Apply hard bounds if requested
            if self.hard_bounds is not None:
                pre_clamping = angles.clone()
                angles = torch.clamp(
                    angles,
                    min=self.hard_bounds[0],
                    max=self.hard_bounds[1],
                )
                self.clamping_error = (pre_clamping - angles).abs().sum().item()
            
            # Keep history if requested
            if self.angle_tracking:
                self.angle_history.append(angles.clone().tolist())
            
            # Check for NaN
            if torch.isnan(angles).any():
                raise ValueError(f"NaN detected in angles: {angles}")
            
            self.angles = angles
            return self.angles.clone()

    def save(self, path: str | Path) -> None:
        """Save learnable parameters to file."""
        path = Path(path)
        to_save = {
            "phase": self.phase.detach().cpu(),
            "w": self.w.detach().cpu(),
            "amplitudes": self.amplitudes.detach().cpu(),
            "ha": self.ha.detach().cpu(),
            "b": self.b.detach().cpu(),
        }
        torch.save(to_save, path)
        console.log(f"[green]Saved parameters to {path}[/green]")

    def load(self, path: str | Path) -> None:
        """Load learnable parameters from file."""
        path = Path(path)
        loaded = torch.load(path, map_location="cpu")
        self.phase.data = loaded["phase"]
        self.w.data = loaded["w"]
        self.amplitudes.data = loaded["amplitudes"]
        self.ha.data = loaded["ha"]
        self.b.data = loaded["b"]
        console.log(f"[green]Loaded parameters from {path}[/green]")


def main() -> None:
    """Example usage of SimpleCPG."""
    adj_dict = create_fully_connected_adjacency(3)
    cpg = SimpleCPG(adj_dict, angle_tracking=True)

    for _ in range(1000):
        cpg.forward()

    import matplotlib.pyplot as plt

    hist = torch.tensor(cpg.angle_history)
    times = torch.arange(hist.shape[0]) * cpg.dt

    plt.figure(figsize=(10, 5))
    for j in range(hist.shape[1]):
        plt.plot(times, hist[:, j], label=f"joint {j}", linewidth=2)
    plt.xlabel("time (s)")
    plt.ylabel("angle (rad)")
    plt.title("SimpleCPG angle histories")
    plt.legend()
    plt.grid(visible=True)
    plt.tight_layout()
    plt.savefig(DATA / "simple_cpg_angles.png")
    plt.show()

    console.log(f"[green]Saved plot to {DATA / 'simple_cpg_angles.png'}[/green]")


if __name__ == "__main__":
    main()
