import numpy as np
import typing as tp
from ..base import ExperimentFunction as ExperimentFunction
from _typeshed import Incomplete

class Agent:
    """An agent has an input size, an output size, a number of layers, a width of its internal layers
    (a.k.a number of neurons per hidden layer)."""
    input_size: Incomplete
    output_size: Incomplete
    layers: Incomplete
    def __init__(self, input_size: int, output_size: int, layers: int = 3, layer_width: int = 14) -> None: ...
    @property
    def dimension(self) -> int: ...
    def set_parameters(self, weights: np.ndarray) -> None: ...
    def get_output(self, data: np.ndarray) -> np.ndarray: ...

class PowerSystem(ExperimentFunction):
    """Very simple model of a power system.
    Real life is more complicated!

    Parameters
    ----------
    num_dams: int
        number of dams to be managed
    depth: int
        number of layers in the neural networks
    width: int
        number of neurons per hidden layer
    year_to_day_ratio: float = 2.
        Ratio between std of consumption in the year and std of consumption in the day.
    constant_to_year_ratio: float
        Ratio between constant baseline consumption and std of consumption in the year.
    back_to_normal: float
        Part of the variability which is forgotten at each time step.
    consumption_noise: float
        Instantaneous variability.
    num_thermal_plants: int
        Number of thermal plants.
    num_years: float
        Number of years.
    failure_cost: float
        Cost of not satisfying the demand. Equivalent to an expensive infinite capacity thermal plant.
    """
    num_dams: Incomplete
    losses: list[float]
    marginal_costs: list[float]
    year_to_day_ratio: Incomplete
    constant_to_year_ratio: Incomplete
    back_to_normal: Incomplete
    consumption_noise: Incomplete
    num_thermal_plants: Incomplete
    number_of_years: Incomplete
    failure_cost: Incomplete
    hydro_prod_per_time_step: list[tp.Any]
    consumption_per_time_step: list[tp.Any]
    average_consumption: Incomplete
    thermal_power_capacity: Incomplete
    thermal_power_prices: Incomplete
    dam_agents: Incomplete
    def __init__(self, num_dams: int = 13, depth: int = 3, width: int = 3, year_to_day_ratio: float = 2.0, constant_to_year_ratio: float = 1.0, back_to_normal: float = 0.5, consumption_noise: float = 0.1, num_thermal_plants: int = 7, num_years: float = 1.0, failure_cost: float = 500.0) -> None: ...
    def get_num_vars(self) -> list[tp.Any]: ...
    def _simulate_power_system(self, *arrays: np.ndarray) -> float: ...
    def make_plots(self, filename: str = 'ps.png') -> None: ...
