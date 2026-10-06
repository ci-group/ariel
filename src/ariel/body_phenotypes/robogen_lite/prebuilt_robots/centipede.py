from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule

SUBMODULES = HingeModule | BrickModule
MODULES = CoreModule | SUBMODULES

F = ModuleFaces.FRONT
L = ModuleFaces.LEFT
R = ModuleFaces.RIGHT


def make_core():
    core = CoreModule(index=0)
    core.name = "C"
    return core


def attach(
    parent: MODULES,
    face: ModuleFaces,
    module: SUBMODULES,
    name: str,
) -> SUBMODULES:
    name = f"{parent.name}-{name}"
    parent.sites[face].attach_body(body=module.body, prefix=name + "-")
    module.name = name
    return module


def add_limbs(block: BrickModule, idx_offset: int) -> None:
    lh = HingeModule(index=idx_offset)
    lh.rotate(45)
    lb = attach(block, L, lh, "LH")
    lb = attach(lh, F, BrickModule(index=idx_offset + 1), "B")
    lkh = HingeModule(index=idx_offset + 2)
    lkh.rotate(90)
    attach(lb, F, lkh, "KH")
    attach(lkh, F, BrickModule(index=idx_offset + 3), "FB")

    rh = HingeModule(index=idx_offset + 4)
    rh.rotate(-45)
    rb = attach(block, R, rh, "RH")
    rb = attach(rh, F, BrickModule(index=idx_offset + 5), "B")
    rkh = HingeModule(index=idx_offset + 6)
    rkh.rotate(135)
    attach(rb, F, rkh, "KH")
    attach(rkh, F, BrickModule(index=idx_offset + 7), "FB")


def body_centipede_n(n_pairs: int) -> CoreModule:
    """Centipede with n_pairs limb-bearing segments on a linear spine."""
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be >= 1, got {n_pairs}")
    core = make_core()

    # Build spine: n_pairs × (HingeModule, BrickModule)
    spine_blocks: list[BrickModule] = []
    prev: MODULES = core
    for i in range(n_pairs):
        h = attach(prev, F, HingeModule(index=2 * i + 1), f"H{i}")
        b = attach(h, F, BrickModule(index=2 * i + 2), f"B{i}")
        spine_blocks.append(b)
        prev = b

    # Attach limbs; each add_limbs call uses 8 consecutive indices
    limb_base = 2 * n_pairs + 1
    for i, block in enumerate(spine_blocks):
        add_limbs(block, idx_offset=limb_base + i * 8)

    return core


def body_centipede() -> CoreModule:
    core = make_core()

    # Spine: Core -> H -> Block1 -> H -> Block2
    h0 = attach(core, F, HingeModule(index=1), "H0")
    b0 = attach(h0, F, BrickModule(index=2), "B0")
    h1 = attach(b0, F, HingeModule(index=3), "H1")
    b1 = attach(h1, F, BrickModule(index=4), "B1")

    # Limbs on Block1: indices 5-12
    add_limbs(b0, idx_offset=5)

    # Limbs on Block2: indices 13-20
    add_limbs(b1, idx_offset=13)

    return core
