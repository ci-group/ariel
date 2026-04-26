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
from typing import NamedTuple

# Third-party libraries
import numpy as np
import torch
from rich.console import Console
from rich.traceback import install
from torch import nn

# Global constants
E = 1e-9

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)


# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(exist_ok=True)

# --- TERMINAL OUTPUT SETUP ---
install(show_locals=False)
console = Console()
torch.set_printoptions(precision=4)


# --- Beta Parameters --- #
class BetaParams(NamedTuple):
    a: float
    b: float
    c: float
    d: float


def create_fully_connected_adjacency(num_nodes: int) -> dict[int, list[int]]:
    adjacency_dict = {}
    for i in range(num_nodes):
        adjacency_dict[i] = [j for j in range(num_nodes) if j != i]
    return adjacency_dict


class NaCPGBeta(nn.Module):
    """Implements the Normalized Asymmetric CPG (NA-CPG)."""

    xy: torch.Tensor
    xy_dot_old: torch.Tensor
    angles: torch.Tensor

    alpha: float
    coefficients: tuple[float, float, float, float] = (
        -80.23623,
        52.11833,
        53.63702,
        72.79792,
    )
    xy_init_value: float = 0.1
    xy_dot_old_init_value: float = 0.1
    phase_params = BetaParams(
        a=1,
        b=1,
        c=0,
        d=2,
    )
    w_params = BetaParams(
        a=1,
        b=1,
        c=0,
        d=2,
    )
    amplitude_params = BetaParams(
        a=1,
        b=1,
        c=0,
        d=2,
    )
    ha_params = BetaParams(
        a=1,
        b=1,
        c=0,
        d=2,
    )
    b_params = BetaParams(
        a=1,
        b=1,
        c=0,
        d=2,
    )

    def __init__(
        self,
        adjacency_dict: dict[int, list[int]],
        alpha: float = 0.1,
        dt: float = 0.1,
        angle_limits: tuple[float, float] | None = (
            -torch.pi / 2,  # -90 degrees
            torch.pi / 2,  # 90 degrees
        ),
        seed: int | None = None,
        *,
        clipping: bool = True,
        angle_tracking: bool = False,
    ) -> None:
        # ================================================================== #
        # User parameters
        # ------------------------------------------------------------------ #
        super().__init__()

        # Connectivity
        self.adjacency_dict = adjacency_dict
        self.n = len(adjacency_dict)

        # Angles
        self.angle_tracking = angle_tracking
        self.angle_history = []
        self.angle_limits = angle_limits
        self.clipping = clipping
        self.clamping_error = 0.0

        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        # ================================================================== #
        # Inherent parameters: do not change during learning
        # ------------------------------------------------------------------ #
        # Learning rate (alpha)
        self.alpha = alpha

        # Time step (dt)
        self.dt = dt

        # ================================================================== #
        # Adaptable parameters
        # ------------------------------------------------------------------ #
        # --- Definitely to adapt --- #
        self.phase = nn.Parameter(
            self.scaled_beta(
                self.phase_params,
                size=(self.n,),
                negative_reflect=False,
            ),
            requires_grad=False,
        )

        self.amplitudes = nn.Parameter(
            self.scaled_beta(
                self.amplitude_params,
                size=(self.n,),
                negative_reflect=False,
            ),
            requires_grad=False,
        )

        # --- Probably to adapt --- #
        # Angular frequency
        self.w = nn.Parameter(
            self.scaled_beta(
                self.w_params,
                size=(self.n,),
                negative_reflect=False,
            ),
            requires_grad=False,
        )

        # --- Probably not to adapt --- #
        # Damping coefficient
        self.ha = nn.Parameter(
            self.scaled_beta(
                self.ha_params,
                size=(self.n,),
                negative_reflect=False,
            ),
            requires_grad=False,
        )

        # Offset ratio of the rhythmic signals
        self.b = nn.Parameter(
            self.scaled_beta(
                self.b_params,
                size=(self.n,),
                negative_reflect=False,
            ),
            requires_grad=False,
        )

        # Store parameters information
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
        # CPG outputs
        self.register_buffer(
            "xy",
            self.init_state(self.xy_init_value, size=(self.n, 2)),
        )

        # Previous CPG outputs (for derivative calculation)
        self.register_buffer(
            "xy_dot_old",
            self.init_state(self.xy_dot_old_init_value, size=(self.n, 2)),
        )

        # CPG angles (results)
        self.register_buffer(
            "angles",
            torch.zeros(self.n),
        )

        # Initial state (for resetting)
        self.initial_state = {
            "xy": self.xy.clone(),
            "xy_dot_old": self.xy_dot_old.clone(),
            "angles": self.angles.clone(),
        }

    def init_state(self, value: float, size: int | list[int]) -> torch.Tensor:
        return torch.Tensor(RNG.choice([-value, value], size=size))

    def scaled_beta(
        self,
        beta_params: BetaParams,
        size: int | list[int],
        *,
        negative_reflect: bool = True,
    ) -> torch.Tensor:
        # Unpack parameters
        a, b, c, d = beta_params

        # a & b must be grater than 0
        lower = min(c, d)
        upper = max(c, d)

        # If negative_reflect is True, c and d must be >= 0
        if negative_reflect and (c < 0 or d < 0):
            msg = "c and d must be non-negative when negative_reflect is True"
            raise ValueError(msg)

        # Generate samples from the scaled beta distribution
        sample = lower + (upper - lower) * RNG.beta(a=a, b=b, size=size)

        # Reflect some values to negative if specified
        if negative_reflect:
            reflection_mask = RNG.choice([-1, 1], size=size)
            return torch.Tensor(sample * reflection_mask)
        return torch.Tensor(sample)

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

    def set_state(
        self,
        xy: list[float] | np.ndarray | torch.Tensor,
        xy_dot_old: list[float] | np.ndarray | torch.Tensor,
    ) -> None:
        self.xy = self.param_type_converter(xy)
        self.xy_dot_old = self.param_type_converter(xy_dot_old)

    def reset_state(self) -> None:
        self.xy.data = self.initial_state["xy"].clone()
        self.xy_dot_old.data = self.initial_state["xy_dot_old"].clone()
        self.angles.data = self.initial_state["angles"].clone()
        self.angle_history = []

    def constraint_function(
        self,
        coefficients: tuple[float, float, float, float],
        w: float,
    ) -> float:
        return float(np.polyval(coefficients, w))

    def run_for(self, steps: int) -> torch.Tensor:
        for _ in range(steps):
            self.forward()
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

    def forward(
        self,
        time: float | None = None,
    ) -> torch.Tensor:
        # Reset if time is zero
        if time is not None and torch.isclose(
            torch.tensor(time),
            torch.tensor(0.0),
        ):
            self.reset_state()

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
                # diff = self.constraint_function(
                #     self.coefficients,
                #     w=self.w.cpu().numpy()[i],
                # )
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

            # Apply angle limits if requested
            if self.angle_limits is not None:
                pre_clamping = angles.clone()
                post_clamping = torch.clamp(
                    angles.clone(),
                    min=self.angle_limits[0],
                    max=self.angle_limits[1],
                )

                # Track how much clamping was done (can be used as a loss)
                self.clamping_error = (
                    (pre_clamping - post_clamping).abs().mean().item()
                )

                # Optionally, apply clipping to the angles
                if self.clipping is True:
                    angles = post_clamping

            # Keep history if requested
            if self.angle_tracking:
                self.angle_history.append(angles.clone().tolist())

        # Check if there are any NaN values in the angle signal
        if np.any(np.isnan(angles.cpu().numpy())):
            msg = "NaN values detected in the angle signal.\n"
            msg += f"{angles.cpu().numpy()=}\n"
            msg += f"{self.clamping_error.cpu().numpy()=}\n"
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


# Example usage
def main() -> None:
    # Imports for the example
    import matplotlib.pyplot as plt

    # Test loop
    losses_clip = []
    losses_curvature = []
    losses_dev = []

    for _ in range(10):  # run fewer times for demo
        adj_dict = create_fully_connected_adjacency(3)
        na_cpg = NaCPGBeta(adj_dict, angle_tracking=True, angle_limits=None)

        for _ in range(100):
            na_cpg.forward()

        # Clipping error
        loss_clip = na_cpg.clamping_error
        losses_clip.append(loss_clip)

        # Flat-line error
        w = np.arange(len(na_cpg.angle_history))
        y = np.mean(na_cpg.angle_history, axis=1)

        # Linear fit
        m, b = np.polyfit(w, y, 1)

        # Deviation from linearity
        dev = np.mean(np.abs(m * w - y + b) / np.sqrt(m * m + 1))
        losses_dev.append(dev)

        # Curvature
        curvature = (
            np.mean(
                (w[2:] - 2 * w[1:-1] + w[:-2]) ** 2
                + (y[2:] - 2 * y[1:-1] + y[:-2]) ** 2,
            )
            if len(w) > 2
            else 0
        )
        losses_curvature.append(curvature)

        hist = torch.tensor(na_cpg.angle_history)
        times = torch.arange(hist.shape[0]) * na_cpg.dt

        plt.figure(figsize=(8, 4))
        for j in range(hist.shape[1]):
            plt.plot(times, hist[:, j], label=f"joint {j}")
        plt.xlabel("time (s)")
        plt.ylabel("angle")
        plt.title("CPG angle histories")
        plt.legend()
        plt.grid(visible=True)
        plt.tight_layout()
        plt.savefig(DATA / "angle_histories.png")
        plt.show()

        # Save learnable parameters
        na_cpg.save(DATA / "na_cpg_params.pt")

    # Final stats
    console.log(f"Loss clipping: {np.mean(losses_clip):.6f}")
    console.log(f"Loss deviation: {np.mean(losses_dev):.6f}")
    console.log(f"Loss curvature: {np.mean(losses_curvature)}")


if __name__ == "__main__":
    """
    Experimentally, the following parameters yield good results:
    alpha: 0.9545
    dt: 0.0643
    coefficients: [-80.23623  52.11833  53.63702  72.79792]
    xy: {'value': -5.91047}
    xy_dot_old: {'value': -8.47188}
    phase: {'a': 2.52254, 'b': 3.76542, 'c': 0.27862, 'd': 0.21873}
    w: {'a': 1.35759, 'b': 0.83381, 'c': 0.41179, 'd': 1.56244}
    amplitudes: {'a': 2.486, 'b': 2.64884, 'c': 0.21248, 'd': 0.4319}
    ha: {'a': 2.53736, 'b': 1.25011, 'c': 0.58488, 'd': 0.82824}
    b: {'a': 2.9718, 'b': 0.56805, 'c': 0.4243, 'd': 0.83754}
    """
    main()
