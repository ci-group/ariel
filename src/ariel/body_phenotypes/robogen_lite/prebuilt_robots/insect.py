from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule


def insect_small() -> CoreModule:
    """Custom robot body built with the 3D editor."""
    core = CoreModule(index=0)
    hinge_0 = HingeModule(index=1)
    core.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_0.body,
        prefix="hinge_0",
    )
    brick_0 = BrickModule(index=4)
    hinge_0.sites[ModuleFaces.FRONT].attach_body(
        body=brick_0.body,
        prefix="brick_0",
    )
    hinge_1 = HingeModule(index=9)
    brick_0.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_1.body,
        prefix="hinge_1",
    )
    brick_1 = BrickModule(index=10)
    hinge_1.sites[ModuleFaces.FRONT].attach_body(
        body=brick_1.body,
        prefix="brick_1",
    )
    hinge_2 = HingeModule(index=53)
    hinge_2.rotate(90)
    brick_1.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_2.body,
        prefix="hinge_2",
    )
    brick_2 = BrickModule(index=54)
    hinge_2.sites[ModuleFaces.FRONT].attach_body(
        body=brick_2.body,
        prefix="brick_2",
    )
    hinge_3 = HingeModule(index=55)
    hinge_3.rotate(-90)
    brick_1.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_3.body,
        prefix="hinge_3",
    )
    brick_3 = BrickModule(index=56)
    hinge_3.sites[ModuleFaces.FRONT].attach_body(
        body=brick_3.body,
        prefix="brick_3",
    )
    hinge_4 = HingeModule(index=49)
    hinge_4.rotate(90)
    brick_0.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_4.body,
        prefix="hinge_4",
    )
    brick_4 = BrickModule(index=50)
    hinge_4.sites[ModuleFaces.FRONT].attach_body(
        body=brick_4.body,
        prefix="brick_4",
    )
    hinge_5 = HingeModule(index=51)
    hinge_5.rotate(-90)
    brick_0.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_5.body,
        prefix="hinge_5",
    )
    brick_5 = BrickModule(index=52)
    hinge_5.sites[ModuleFaces.FRONT].attach_body(
        body=brick_5.body,
        prefix="brick_5",
    )
    hinge_6 = HingeModule(index=33)
    hinge_6.rotate(-90)
    core.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_6.body,
        prefix="hinge_6",
    )
    brick_6 = BrickModule(index=34)
    hinge_6.sites[ModuleFaces.FRONT].attach_body(
        body=brick_6.body,
        prefix="brick_6",
    )
    hinge_7 = HingeModule(index=43)
    hinge_7.rotate(90)
    core.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_7.body,
        prefix="hinge_7",
    )
    brick_7 = BrickModule(index=44)
    hinge_7.sites[ModuleFaces.FRONT].attach_body(
        body=brick_7.body,
        prefix="brick_7",
    )
    return core


# --------------------------------------------------------------------------- #
# Parametric variants used for scaling studies.
#
# Foot bricks are tagged with the name "FB" so their final mujoco body names
# end in "-FB-brick", matching the foot-discovery convention used by the
# centipede morphology (see experiment/rl/undirected_locomotion_ppo_jax.py).
# --------------------------------------------------------------------------- #

SUBMODULES = HingeModule | BrickModule
MODULES = CoreModule | SUBMODULES

_F = ModuleFaces.FRONT
_L = ModuleFaces.LEFT
_R = ModuleFaces.RIGHT


def _attach(parent: MODULES, face: ModuleFaces, module: SUBMODULES, name: str) -> SUBMODULES:
    full = f"{parent.name}-{name}" if getattr(parent, "name", None) else name
    parent.sites[face].attach_body(body=module.body, prefix=full + "-")
    module.name = full
    return module


def _add_leg_pair(
    block: MODULES, start_idx: int, right_rot: float, left_rot: float,
) -> int:
    """Attach a L/R hinge-plus-foot pair to ``block``; returns next free index."""
    rh = HingeModule(index=start_idx); rh.rotate(right_rot)
    _attach(block, _R, rh, "RH")
    _attach(rh, _F, BrickModule(index=start_idx + 1), "FB")

    lh = HingeModule(index=start_idx + 2); lh.rotate(left_rot)
    _attach(block, _L, lh, "LH")
    _attach(lh, _F, BrickModule(index=start_idx + 3), "FB")
    return start_idx + 4


def insect_n(n_segments: int) -> CoreModule:
    """Insect with a scalable forward spine; the core and every spine brick
    carry one L/R leg pair.

    ``legs = 2 * (n_segments + 1)``. ``n_segments=2`` matches the leg count
    and topology of :func:`insect_small` (6 legs, 2 spine segments).
    """
    if n_segments < 0:
        raise ValueError(f"n_segments must be >= 0, got {n_segments}")

    core = CoreModule(index=0)
    core.name = "C"
    idx = 1

    spine_bricks: list[BrickModule] = []
    prev: MODULES = core
    for i in range(n_segments):
        h = HingeModule(index=idx); idx += 1
        _attach(prev, _F, h, f"H{i}")
        b = BrickModule(index=idx); idx += 1
        _attach(h, _F, b, f"B{i}")
        spine_bricks.append(b)
        prev = b

    # Core legs: right hinge rotates -90, left +90 (as in insect_small).
    idx = _add_leg_pair(core, idx, right_rot=-90, left_rot=90)
    # Spine-brick legs: right hinge rotates +90, left -90 (as in insect_small).
    for block in spine_bricks:
        idx = _add_leg_pair(block, idx, right_rot=90, left_rot=-90)

    return core


def insect_longspine_n(n_segments: int) -> CoreModule:
    """Scale spine length while keeping legs only on the core and the terminal
    spine brick. Fixed leg count (4 legs, 2 pairs); the action dimension grows
    only through added spine hinges, isolating spine-mass/length effects from
    leg-count scaling.
    """
    if n_segments < 1:
        raise ValueError(f"n_segments must be >= 1, got {n_segments}")

    core = CoreModule(index=0)
    core.name = "C"
    idx = 1

    prev: MODULES = core
    tip: BrickModule | None = None
    for i in range(n_segments):
        h = HingeModule(index=idx); idx += 1
        _attach(prev, _F, h, f"H{i}")
        b = BrickModule(index=idx); idx += 1
        _attach(h, _F, b, f"B{i}")
        prev = b
        tip = b

    idx = _add_leg_pair(core, idx, right_rot=-90, left_rot=90)
    assert tip is not None
    idx = _add_leg_pair(tip, idx, right_rot=90, left_rot=-90)

    return core