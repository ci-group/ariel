"""Command-line entry point for the spatial evolutionary algorithm.

Flags are generated from the configuration model, so every setting is
overridable and the parser cannot drift out of step with
:class:`~ariel.spatial_ea.config.SpatialEAConfig`.
"""

# Standard library
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, get_args, get_origin

# Local libraries
from ariel import console
from ariel.spatial_ea.config import SpatialEAConfig
from ariel.spatial_ea.engine import SpatialEA

# Global constants
TUPLE_FIELDS = frozenset({"world_size", "mating_zone_center"})


def _literal_choices(annotation: Any) -> list[str] | None:
    """Extract the allowed values of a ``Literal`` annotation.

    Parameters
    ----------
    annotation
        A field annotation, possibly an alias for a ``Literal``.

    Returns
    -------
        The permitted string values, or ``None`` if the annotation is not a
        ``Literal`` of strings.
    """
    target = getattr(annotation, "__value__", annotation)
    if get_origin(target) is not None and str(get_origin(target)).endswith(
        "Literal",
    ):
        values = get_args(target)
        if all(isinstance(value, str) for value in values):
            return list(values)
    return None


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser from the configuration model.

    Returns
    -------
        A parser with ``--config`` plus one flag per configuration field.
    """
    parser = argparse.ArgumentParser(
        prog="python -m ariel.spatial_ea",
        description="Run the ARIEL spatial evolutionary algorithm.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML configuration file; native or legacy schema.",
    )

    for name, info in SpatialEAConfig.model_fields.items():
        flag = f"--{name.replace('_', '-')}"
        annotation = info.annotation
        kwargs: dict[str, Any] = {"default": None, "dest": name}

        choices = _literal_choices(annotation)
        if choices is not None:
            kwargs["type"] = str
            kwargs["choices"] = choices
        elif annotation is bool:
            kwargs["type"] = lambda value: value.lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            kwargs["metavar"] = "BOOL"
        elif annotation is int:
            kwargs["type"] = int
        elif annotation is float:
            kwargs["type"] = float
        elif annotation is Path:
            kwargs["type"] = Path
        elif name in TUPLE_FIELDS:
            kwargs["type"] = float
            kwargs["nargs"] = 2
        else:
            kwargs["type"] = str

        parser.add_argument(flag, **kwargs)

    return parser


def config_from_args(args: argparse.Namespace) -> SpatialEAConfig:
    """Build a configuration from parsed arguments.

    A ``--config`` file provides the base, and any explicitly passed flag
    overrides the corresponding field.

    Parameters
    ----------
    args
        Parsed command-line arguments.

    Returns
    -------
        The resolved configuration.
    """
    if args.config is not None:
        config = SpatialEAConfig.from_yaml(args.config)
    else:
        config = SpatialEAConfig()

    overrides = {
        name: (
            tuple(getattr(args, name))
            if name in TUPLE_FIELDS
            else getattr(args, name)
        )
        for name in SpatialEAConfig.model_fields
        if getattr(args, name, None) is not None
    }

    if overrides:
        config = config.model_copy(update=overrides)

    return config


def main(argv: list[str] | None = None) -> int:
    """Run one spatial EA from the command line.

    Parameters
    ----------
    argv
        Argument list, defaulting to ``sys.argv[1:]``.

    Returns
    -------
        ``0`` when a best individual was found, ``1`` when the population
        died out.
    """
    args = build_parser().parse_args(argv)
    config = config_from_args(args)

    engine = SpatialEA(config=config)
    best = engine.run()

    if best is None:
        console.print("[bold red]Spatial EA finished with no survivors[/]")
        return 1

    console.print("[bold green]Spatial EA complete[/]")
    console.print(
        f"Best individual: id={best.unique_id} "
        f"fitness={best.fitness:.6f} generation={best.generation}",
    )
    console.print(f"Final population size: {len(engine.population)}")
    return 0
