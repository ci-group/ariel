"""Compile all terrains to XML files."""

# Standard library
import inspect

# Local libraries
import ariel.simulation.environments as envs
from ariel import console


def compile_all_world(*, with_load_compiled: bool = True) -> None:
    """Entry point."""
    for name, cls in inspect.getmembers(envs, inspect.isclass):
        world: envs.BaseWorld = cls(load_precompiled=with_load_compiled)
        world.store_to_xml()
        console.print(f"Compiled '{name}' to XML.")


if __name__ == "__main__":
    compile_all_world()
