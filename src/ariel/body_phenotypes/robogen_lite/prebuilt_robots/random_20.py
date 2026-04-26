"""Randomized 20-module prebuilt morphology.

This module exposes a reproducible random morphology generator that creates
exactly 20 modules (1 core + 19 submodules), making it larger than the
standard spider prebuilt body.
"""

import random

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule


def random_spider20(seed: int = 1337) -> CoreModule:
    """Build a reproducible random 20-module morphology.

    Parameters
    ----------
    seed
        Seed for reproducible random morphology generation.

    Returns
    -------
    CoreModule
        Core of a randomized robot with exactly 20 total modules.
    """
    rng = random.Random(seed)

    core = CoreModule(index=0)
    seed_faces = [
        ModuleFaces.FRONT,
        ModuleFaces.LEFT,
        ModuleFaces.BACK,
        ModuleFaces.RIGHT,
    ]

    # Queue of still-available attachment sites: (parent_module, parent_face)
    open_sites: list[tuple[CoreModule | HingeModule | BrickModule, ModuleFaces]] = [
        (core, face) for face in seed_faces if face in core.sites
    ]

    hinge_rotations = [0, 45, 90]

    # Create 19 submodules so total modules (including core) is exactly 20.
    for idx in range(1, 20):
        if not open_sites:
            raise RuntimeError("No open attachment sites remain while building random_spider20")

        site_idx = rng.randrange(len(open_sites))
        parent, attach_face = open_sites.pop(site_idx)

        # Slight hinge bias usually increases articulation in random bodies.
        if rng.random() < 0.6:
            child = HingeModule(index=idx)
            child.rotate(rng.choice(hinge_rotations))
        else:
            child = BrickModule(index=idx)

        parent.sites[attach_face].attach_body(
            body=child.body,
            prefix=f"m{idx}",
        )

        # Only enqueue attachment faces that this module actually exposes.
        for child_face in child.sites.keys():
            open_sites.append((child, child_face))

    return core
