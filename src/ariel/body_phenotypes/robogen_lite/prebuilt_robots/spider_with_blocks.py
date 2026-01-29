import math
import sys
from typing import Callable

import numpy as np

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule

current_module = sys.modules[__name__]

SUBMODULES = HingeModule | BrickModule
MODULES = CoreModule | SUBMODULES

faces = [
    ModuleFaces.FRONT, ModuleFaces.LEFT, ModuleFaces.BACK, ModuleFaces.RIGHT,
]
F, L, B, R = faces

def make_core():
    core = CoreModule(index=0)
    core.name = "C"
    return core

def attach(parent: MODULES,
           face: ModuleFaces,
           module: SUBMODULES,
           name: str,
           rotation: float = 0) -> SUBMODULES:

    name = f"{parent.name}-{name}"
    parent.sites[face].attach_body(body=module.body, prefix=name + "-")

    module.name = name

    if rotation != 0:
        module.rotate(rotation)

    return module

def body_spider() -> CoreModule:
    core = make_core()

    for i, f in enumerate(faces, start=1):
        h0 = attach(core, f, HingeModule(index=i), f"{f.name[0]}H")
        b0 = attach(h0, F, BrickModule(index=4+i), "B")
        h1 = attach(b0, F, HingeModule(index=8+i), "H", rotation=90)
        b1 = attach(h1, F, BrickModule(index=12+i), "B")

    return core

def body_spider45() -> CoreModule:
    core = body_spider()
    core.spec.body("core").quat = (math.cos(math.pi / 8), 0, 0, math.sin(math.pi / 8))
    return core
