"""
TODO(jmdm): description of script.

Notes
-----
    *

References
----------
    [1] https://www.sciencedirect.com/science/article/pii/S2667379722000353

Todo
----
    [ ] Fix constraint function:
        This requires experimental validation to find the mapping from angular
        velocity to maximum allowed change in the CPG state space.
        The paper determines this information empirically.
    [ ] Implement matrix formulation
    [ ] What should the initial values be???
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
    adjacency_dict = {}
    for i in range(num_nodes):
        adjacency_dict[i] = [j for j in range(num_nodes) if j != i]
    return adjacency_dict


class NaCPG_Norm(nn.Module):
    """Implements the Normalized Asymmetric CPG (NA-CPG)."""

    raise NotImplementedError(
        "This class is a work in progress and not ready for use."
    )

    xy: torch.Tensor
    xy_dot_old: torch.Tensor
    angles: torch.Tensor

    def __init__(
        self,
        adjacency_dict: dict[int, list[int]],
        alpha: float = 0.1,
        dt: float = 0.02,
        hard_bounds: tuple[float, float] | None = (-torch.pi / 2, torch.pi / 2),
        *,
        angle_tracking: bool = False,
        normalize: bool = False,
        duration: int = 200,
        seed: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        adjacency_dict : dict[int, list[int]]
            Dictionary defining the connectivity of the CPG network.
            Each key is a node index, and the value is a list of indices
            of connected nodes.
        alpha : float, optional
            Learning rate for the CPG dynamics, by default 0.1
        dt : float, optional
            Time step for the CPG updates, by default 0.01
        hard_bounds : tuple[float, float] | None, optional
            Tuple defining the minimum and maximum angle bounds.
            If None, no bounds are applied, by default (-π/2, π/2)
        angle_tracking : bool, optional
            If True, stores the history of angles for analysis,
            by default False
        normalize : bool, optional
            If True, normalizes the output angles to [-π/2, π/2],
            by default False
        duration : int, optional
            Duration in seconds for which the CPG will run
        seed : int | None, optional
            Random seed for reproducibility, by default None
        """
        # ================================================================== #
        # User parameters
        # ------------------------------------------------------------------ #
        super().__init__()
        self.adjacency_dict = adjacency_dict
        self.n = len(adjacency_dict)
        self.angle_tracking = angle_tracking
        self.hard_bounds = hard_bounds
        self.clamping_error = 0.0
        self.normalize = normalize
        self.duration = duration
        if seed is not None:
            torch.manual_seed(seed)

        # ================================================================== #
        # Inherent parameters: do not change during learning
        # ------------------------------------------------------------------ #
        # Learning rate (alpha)
        self.alpha = alpha

        # Time step (dt)
        self.dt = dt

        # Step counter
        self.current_step = 0

        # ================================================================== #
        # Adaptable parameters
        # ------------------------------------------------------------------ #
        scale = torch.pi * 2
        # --- Definitely to adapt --- #
        self.phase = nn.Parameter(
            ((torch.rand(self.n) * 2 - 1) * scale),
            requires_grad=False,
        )
        self.amplitudes = nn.Parameter(
            ((torch.rand(self.n) * 2 - 1) * scale),
            requires_grad=False,
        )

        # --- Probably to adapt --- #
        self.w = nn.Parameter(
            ((torch.rand(self.n) * 2 - 1) * scale),
            requires_grad=False,
        )

        # --- Probably not to adapt --- #
        self.ha = nn.Parameter(
            torch.randn(self.n),
            requires_grad=False,
        )
        self.b = nn.Parameter(
            torch.randint(
                -100,
                100,
                (self.n,),
                dtype=torch.float32,
            ),
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
        # Internal states (buffers, not learnable)
        # ------------------------------------------------------------------ #
        self.register_buffer(
            "xy",
            torch.randn(self.n, 2),
        )
        self.register_buffer(
            "xy_dot_old",
            torch.randn(self.n, 2),
        )
        self.register_buffer(
            "angles",
            torch.zeros(self.n),
        )

        self.initial_state = {
            "xy": self.xy.clone(),
            "xy_dot_old": self.xy_dot_old.clone(),
            "angles": self.angles.clone(),
        }

        # History of angles
        self.angle_history = []
        self.is_normalized = False
        if self.normalize:
            self.init_angle_history()

    def init_angle_history(self) -> None:
        # console.log("[yellow]Normalizing CPG output angles[/yellow]: Angle tracking enabled since its needed")
        self.angle_tracking = True
        for _ in range(self.duration * int(1 / self.dt) + 10):
            self.forward()
        hist = torch.tensor(self.angle_history)

        # Normalize each joint's signal independently
        self.angle_history = (
            hist / hist.abs().max(dim=0, keepdim=True).values * (torch.pi / 2)
        )
        console.log(f"len angle history {len(self.angle_history)}")
        # console.log(self.angle_history)
        self.is_normalized = True

    def param_type_converter(
        self,
        params: list[float] | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(params, list):
            params = torch.tensor(params, dtype=torch.float32)
        elif isinstance(params, np.ndarray):
            params = torch.from_numpy(params).float()
        return params

    def set_flat_params(self, params: torch.Tensor) -> None:
        # Convert params to tensor if needed
        safe_params = self.param_type_converter(params)

        # Check size is correct
        if safe_params.numel() != self.num_of_parameters:
            msg = "Parameter vector has incorrect size. "
            msg += (
                f"Expected {self.num_of_parameters}, got {safe_params.numel()}."
            )
            raise ValueError(msg)

        # Set parameters
        pointer = 0
        for param in self.parameter_groups.values():
            num_param = param.numel()
            param.data = params[pointer : pointer + num_param].view_as(param)
            pointer += num_param

    def set_param_with_dict(self, params: dict[str, torch.Tensor]) -> None:
        for key, value in params.items():
            safe_value = self.param_type_converter(value)
            self.set_params_by_group(key, safe_value)

    def set_params_by_group(
        self,
        group_name: str,
        params: torch.Tensor,
    ) -> None:
        # Convert params to tensor if needed
        safe_params = self.param_type_converter(params)

        # Check group exists
        if group_name not in self.parameter_groups:
            msg = f"Parameter group '{group_name}' does not exist."
            raise ValueError(msg)

        # Get the parameter group
        param = self.parameter_groups[group_name]
        if safe_params.numel() != param.numel():
            msg = (
                f"Parameter vector has incorrect size for group '{group_name}'."
            )
            raise ValueError(
                msg,
            )
        param.data = safe_params.view_as(param)

    def get_flat_params(self) -> torch.Tensor:
        return torch.cat([p.flatten() for p in self.parameter_groups.values()])

    @staticmethod
    def term_a(alpha: float, r2i: float) -> float:
        return alpha * (1 - r2i**2)

    @staticmethod
    def term_b(zeta_i: float, w_i: float) -> float:
        return (1 / (zeta_i + E)) * w_i

    @staticmethod
    def zeta(ha_i: float, x_dot_old: float) -> float:
        return 1 - ha_i * ((x_dot_old + E) / (torch.abs(x_dot_old) + E))

    def reset(self) -> None:
        self.xy.data = self.initial_state["xy"].clone()
        self.xy_dot_old.data = self.initial_state["xy_dot_old"].clone()
        self.angles.data = self.initial_state["angles"].clone()
        self.angle_history = self.init_angle_history()
        self.current_step = 0

    def step(self, time: float) -> torch.Tensor:
        self.current_step += 1
        # print(len(self.angle_history))
        # print(time)
        return self.angle_history[self.current_step - 1]

    def forward(self, time: float | None = None) -> torch.Tensor:
        # Reset if time is zero
        # if time is not None and torch.isclose(
        #     torch.tensor(time),
        #     torch.tensor(0.0),
        # ):
        #     self.reset()
        # if time is not None and abs(float(time)) < 1e-12:
        #     self.reset()
        # Normalize amplitudes
        if self.normalize and self.is_normalized:
            return self.step(self.current_step)

        # Update CPG states
        with torch.inference_mode():
            # R matrix
            r_matrix = torch.zeros(self.n, self.n, 2, 2)
            for i in range(self.n):
                for j in range(self.n):
                    if i == j:
                        r_matrix[i, j] = torch.eye(2)
                    else:
                        phase_diff_ij = self.phase[i] - self.phase[j]
                        cos_d_ij = torch.cos(phase_diff_ij)
                        sin_d_ij = torch.sin(phase_diff_ij)
                        r_matrix[i, j] = torch.tensor([
                            [cos_d_ij, -sin_d_ij],
                            [sin_d_ij, cos_d_ij],
                        ])

            # K matrix
            k_matrix = torch.zeros(self.n, 2, 2)
            for i in range(self.n):
                x_dot_old, _ = self.xy_dot_old[i]
                ha_i = self.ha[i]
                w_i = self.w[i]
                xi, yi = self.xy[i]

                r2i = xi**2 + yi**2
                term_a = self.term_a(self.alpha, r2i)

                zeta_i = self.zeta(ha_i, x_dot_old)
                term_b = self.term_b(zeta_i, w_i)

                k_matrix[i] = torch.tensor([
                    [term_a, -term_b],
                    [term_b, term_a],
                ])

            # Update each CPG
            angles = torch.zeros(self.n)
            for i, (xi, yi) in enumerate(self.xy):
                # term_a contribution
                term_a_vec = torch.mv(k_matrix[i], self.xy[i])

                # term_b contribution
                term_b_vec = torch.zeros(2)
                for j in self.adjacency_dict[i]:
                    term_b_vec += torch.mv(r_matrix[i, j], self.xy[j])

                # Combine contributions to get the derivative
                xi_dot, yi_dot = term_a_vec + term_b_vec

                # Constraint function (CF)
                xi_dot_old, yi_dot_old = self.xy_dot_old[i]
                diff = 10
                xi_dot = torch.clamp(
                    xi_dot,
                    xi_dot_old - diff,
                    xi_dot_old + diff,
                )
                yi_dot = torch.clamp(
                    yi_dot,
                    yi_dot_old - diff,
                    yi_dot_old + diff,
                )

                # Compute new states
                xi_new = xi + (xi_dot * self.dt)
                yi_new = yi + (yi_dot * self.dt)

                # Save new values
                self.xy_dot_old[i] = self.xy[i]
                self.xy[i] = torch.tensor([xi_new, yi_new])

                # Save the angles (results)
                angles[i] = self.amplitudes[i] * yi_new

            # # Apply hard bounds if requested
            # if self.hard_bounds is not None:
            #     pre_clamping = angles.clone()
            #     angles = torch.clamp(
            #         angles,
            #         min=self.hard_bounds[0],
            #         max=self.hard_bounds[1],
            #         out=angles,
            #     )

            # # Track how much clamping was done (can be used as a loss)
            # self.clamping_error = (pre_clamping - angles).abs().sum().item()

            # Keep history if requested
            if self.angle_tracking:
                self.angle_history.append(angles.clone().tolist())

        # Check if there are any NaN values in the angle signal
        if np.any(np.isnan(angles.cpu().numpy())):
            msg = "NaN values detected in the angle signal.\n"
            msg += f"{angles.cpu().numpy()=}\n"
            # msg += f"{self.clamping_error.cpu().numpy()=}\n"
            msg += f"{self.xy.cpu().numpy()=}\n"
            msg += f"{self.xy_dot_old.cpu().numpy()=}\n"
            msg += f"{self.ha.cpu().numpy()=}\n"
            msg += f"{self.w.cpu().numpy()=}\n"
            msg += f"{self.amplitudes.cpu().numpy()=}\n"
            msg += f"{self.phase.cpu().numpy()=}\n"
            raise ValueError(msg)

        # Save and return the angles
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


# Example usage
def main(run_id: int) -> torch.Tensor:
    dur = 10  # duration in seconds
    t = time.perf_counter()
    adj_dict = create_fully_connected_adjacency(3)
    na_cpg_mat = NaCPG_Norm(
        adj_dict, angle_tracking=True, normalize=True, duration=dur, alpha=30
    )

    range_ = int(dur * int(1 / na_cpg_mat.dt))
    console.log(f"Running NaCPG_Norm for {range_} steps (Run {run_id})")
    for time_step in range(range_):
        na_cpg_mat.forward(time_step)

    hist = torch.tensor(na_cpg_mat.angle_history)
    console.log(
        f"NaCPG_Norm Run {run_id} took {time.perf_counter() - t:.2f} seconds"
    )
    return hist


def plot_all_runs(histories: list[torch.Tensor], dt: float) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    for i, hist in enumerate(histories):
        times = torch.arange(hist.shape[0]) * dt
        plt.subplot(2, 3, i + 1)
        for j in range(hist.shape[1]):
            plt.plot(times, hist[:, j], label=f"joint {j}")
        plt.xlabel("time (s)")
        plt.ylabel("angle")
        plt.title(f"Run {i + 1}")
        plt.legend()
        plt.grid(visible=True)
    plt.tight_layout()
    plt.savefig(DATA / "all_angle_histories.png")
    plt.show()


if __name__ == "__main__":
    import time

    histories = []
    for run_id in range(6):  # Run 6 times
        histories.append(main(run_id))

    # Plot all runs
    plot_all_runs(histories, dt=0.02)
