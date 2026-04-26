"""Test: initialization of config classes."""

# Local libraries
from ariel.parameters import ArielConfig
from ariel.parameters.ariel_modules import SER0019, ArielModulesConfig
from ariel.parameters.mujoco_params import MujocoConfig


def test_ariel_config_initialization() -> None:
    """Simply instantiate the ArielConfig class."""
    ArielConfig()


def test_ariel_modules_config_initialization() -> None:
    """Simply instantiate the ArielModulesConfig class."""
    ArielModulesConfig()


def test_ser0019_initialization() -> None:
    """Simply instantiate the SER0019 class."""
    SER0019()


def test_mujoco_config_initialization() -> None:
    """Simply instantiate the MujocoConfig class."""
    MujocoConfig()
