"""The baby robot configuration."""

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule


def baby_robot() -> CoreModule:
    """Baby robot body built with the 3D editor.

    return: core module of the baby robot body
    """
    core = CoreModule(index=0)
    hinge_0 = HingeModule(index=1)
    core.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_0.body,
        prefix="hinge_0",
    )
    brick_0 = BrickModule(index=4)
    hinge_0.sites[ModuleFaces.FRONT].attach_body(
        body=brick_0.body,
        prefix="brick_0",
    )
    hinge_1 = HingeModule(index=2)
    core.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_1.body,
        prefix="hinge_1",
    )
    brick_1 = BrickModule(index=17)
    hinge_1.sites[ModuleFaces.FRONT].attach_body(
        body=brick_1.body,
        prefix="brick_1",
    )
    hinge_2 = HingeModule(index=3)
    core.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_2.body,
        prefix="hinge_2",
    )
    hinge_3 = HingeModule(index=5)
    hinge_2.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_3.body,
        prefix="hinge_3",
    )
    brick_2 = BrickModule(index=6)
    hinge_3.sites[ModuleFaces.FRONT].attach_body(
        body=brick_2.body,
        prefix="brick_2",
    )
    hinge_4 = HingeModule(index=7)
    brick_2.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_4.body,
        prefix="hinge_4",
    )
    brick_3 = BrickModule(index=8)
    hinge_4.sites[ModuleFaces.FRONT].attach_body(
        body=brick_3.body,
        prefix="brick_3",
    )
    hinge_5 = HingeModule(index=9)
    core.sites[ModuleFaces.BACK].attach_body(
        body=hinge_5.body,
        prefix="hinge_5",
    )
    brick_4 = BrickModule(index=10)
    hinge_5.sites[ModuleFaces.FRONT].attach_body(
        body=brick_4.body,
        prefix="brick_4",
    )
    hinge_6 = HingeModule(index=11)
    brick_4.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_6.body,
        prefix="hinge_6",
    )
    brick_5 = BrickModule(index=12)
    hinge_6.sites[ModuleFaces.FRONT].attach_body(
        body=brick_5.body,
        prefix="brick_5",
    )
    hinge_7 = HingeModule(index=13)
    brick_5.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_7.body,
        prefix="hinge_7",
    )
    brick_6 = BrickModule(index=15)
    hinge_7.sites[ModuleFaces.FRONT].attach_body(
        body=brick_6.body,
        prefix="brick_6",
    )
    hinge_8 = HingeModule(index=14)
    brick_5.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_8.body,
        prefix="hinge_8",
    )
    brick_7 = BrickModule(index=16)
    hinge_8.sites[ModuleFaces.FRONT].attach_body(
        body=brick_7.body,
        prefix="brick_7",
    )
    return core
