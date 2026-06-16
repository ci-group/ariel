"""
Robots from Fuda's dataset, some robots adjusted to fit new ARIEL modules.

The ones that are adjusted have "modified" in the name.

The rest are by me!!! John.
"""

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule


def baby_a() -> CoreModule:
    """Custom robot body built with the 3D editor."""
    core = CoreModule(index=0)
    hinge_0 = HingeModule(index=2)
    hinge_0.rotate(90)
    core.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_0.body,
        prefix="hinge_0",
    )
    brick_0 = BrickModule(index=5)
    hinge_0.sites[ModuleFaces.FRONT].attach_body(
        body=brick_0.body,
        prefix="brick_0",
    )
    hinge_1 = HingeModule(index=3)
    hinge_1.rotate(90)
    core.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_1.body,
        prefix="hinge_1",
    )
    hinge_2 = HingeModule(index=6)
    hinge_2.rotate(90)
    hinge_1.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_2.body,
        prefix="hinge_2",
    )
    brick_1 = BrickModule(index=7)
    hinge_2.sites[ModuleFaces.FRONT].attach_body(
        body=brick_1.body,
        prefix="brick_1",
    )
    hinge_3 = HingeModule(index=9)
    hinge_3.rotate(90)
    brick_1.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_3.body,
        prefix="hinge_3",
    )
    brick_2 = BrickModule(index=10)
    hinge_3.sites[ModuleFaces.FRONT].attach_body(
        body=brick_2.body,
        prefix="brick_2",
    )
    hinge_4 = HingeModule(index=28)
    core.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_4.body,
        prefix="hinge_4",
    )
    brick_3 = BrickModule(index=29)
    hinge_4.sites[ModuleFaces.FRONT].attach_body(
        body=brick_3.body,
        prefix="brick_3",
    )
    hinge_5 = HingeModule(index=30)
    brick_3.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_5.body,
        prefix="hinge_5",
    )
    brick_4 = BrickModule(index=31)
    hinge_5.sites[ModuleFaces.FRONT].attach_body(
        body=brick_4.body,
        prefix="brick_4",
    )
    hinge_6 = HingeModule(index=33)
    hinge_6.rotate(90)
    brick_4.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_6.body,
        prefix="hinge_6",
    )
    brick_5 = BrickModule(index=36)
    hinge_6.sites[ModuleFaces.FRONT].attach_body(
        body=brick_5.body,
        prefix="brick_5",
    )
    hinge_7 = HingeModule(index=172)
    hinge_7.rotate(-90)
    brick_4.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_7.body,
        prefix="hinge_7",
    )
    brick_6 = BrickModule(index=173)
    hinge_7.sites[ModuleFaces.FRONT].attach_body(
        body=brick_6.body,
        prefix="brick_6",
    )
    return core


def baby_b() -> CoreModule:
    """Custom robot body built with the 3D editor."""
    core = CoreModule(index=0)
    hinge_0 = HingeModule(index=2)
    core.sites[ModuleFaces.BACK].attach_body(
        body=hinge_0.body,
        prefix="hinge_0",
    )
    brick_0 = BrickModule(index=8)
    hinge_0.sites[ModuleFaces.FRONT].attach_body(
        body=brick_0.body,
        prefix="brick_0",
    )
    hinge_1 = HingeModule(index=11)
    hinge_1.rotate(90)
    brick_0.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_1.body,
        prefix="hinge_1",
    )
    brick_1 = BrickModule(index=18)
    hinge_1.sites[ModuleFaces.FRONT].attach_body(
        body=brick_1.body,
        prefix="brick_1",
    )
    hinge_2 = HingeModule(index=21)
    hinge_2.rotate(90)
    brick_1.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_2.body,
        prefix="hinge_2",
    )
    brick_2 = BrickModule(index=24)
    hinge_2.sites[ModuleFaces.FRONT].attach_body(
        body=brick_2.body,
        prefix="brick_2",
    )
    hinge_3 = HingeModule(index=3)
    core.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_3.body,
        prefix="hinge_3",
    )
    brick_3 = BrickModule(index=9)
    hinge_3.sites[ModuleFaces.FRONT].attach_body(
        body=brick_3.body,
        prefix="brick_3",
    )
    hinge_4 = HingeModule(index=12)
    hinge_4.rotate(90)
    brick_3.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_4.body,
        prefix="hinge_4",
    )
    brick_4 = BrickModule(index=19)
    hinge_4.sites[ModuleFaces.FRONT].attach_body(
        body=brick_4.body,
        prefix="brick_4",
    )
    hinge_5 = HingeModule(index=20)
    hinge_5.rotate(90)
    brick_4.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_5.body,
        prefix="hinge_5",
    )
    brick_5 = BrickModule(index=25)
    hinge_5.sites[ModuleFaces.FRONT].attach_body(
        body=brick_5.body,
        prefix="brick_5",
    )
    hinge_6 = HingeModule(index=4)
    core.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_6.body,
        prefix="hinge_6",
    )
    brick_6 = BrickModule(index=6)
    hinge_6.sites[ModuleFaces.FRONT].attach_body(
        body=brick_6.body,
        prefix="brick_6",
    )
    hinge_7 = HingeModule(index=5)
    core.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_7.body,
        prefix="hinge_7",
    )
    brick_7 = BrickModule(index=7)
    hinge_7.sites[ModuleFaces.FRONT].attach_body(
        body=brick_7.body,
        prefix="brick_7",
    )
    hinge_8 = HingeModule(index=15)
    hinge_8.rotate(90)
    brick_7.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_8.body,
        prefix="hinge_8",
    )
    brick_8 = BrickModule(index=17)
    hinge_8.sites[ModuleFaces.FRONT].attach_body(
        body=brick_8.body,
        prefix="brick_8",
    )
    hinge_9 = HingeModule(index=22)
    hinge_9.rotate(90)
    brick_8.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_9.body,
        prefix="hinge_9",
    )
    brick_9 = BrickModule(index=23)
    hinge_9.sites[ModuleFaces.FRONT].attach_body(
        body=brick_9.body,
        prefix="brick_9",
    )
    return core


def gecko() -> CoreModule:
    """Custom robot body built with the 3D editor."""
    core = CoreModule(index=0)
    hinge_0 = HingeModule(index=4)
    core.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_0.body,
        prefix="hinge_0",
    )
    brick_0 = BrickModule(index=6)
    hinge_0.sites[ModuleFaces.FRONT].attach_body(
        body=brick_0.body,
        prefix="brick_0",
    )
    hinge_1 = HingeModule(index=44)
    brick_0.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_1.body,
        prefix="hinge_1",
    )
    brick_1 = BrickModule(index=45)
    hinge_1.sites[ModuleFaces.FRONT].attach_body(
        body=brick_1.body,
        prefix="brick_1",
    )
    hinge_2 = HingeModule(index=51)
    hinge_2.rotate(90)
    brick_1.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_2.body,
        prefix="hinge_2",
    )
    brick_2 = BrickModule(index=52)
    hinge_2.sites[ModuleFaces.FRONT].attach_body(
        body=brick_2.body,
        prefix="brick_2",
    )
    hinge_3 = HingeModule(index=53)
    hinge_3.rotate(90)
    brick_1.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_3.body,
        prefix="hinge_3",
    )
    brick_3 = BrickModule(index=54)
    hinge_3.sites[ModuleFaces.FRONT].attach_body(
        body=brick_3.body,
        prefix="brick_3",
    )
    hinge_4 = HingeModule(index=28)
    hinge_4.rotate(90)
    core.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_4.body,
        prefix="hinge_4",
    )
    brick_4 = BrickModule(index=29)
    hinge_4.sites[ModuleFaces.FRONT].attach_body(
        body=brick_4.body,
        prefix="brick_4",
    )
    hinge_5 = HingeModule(index=30)
    hinge_5.rotate(90)
    core.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_5.body,
        prefix="hinge_5",
    )
    brick_5 = BrickModule(index=31)
    hinge_5.sites[ModuleFaces.FRONT].attach_body(
        body=brick_5.body,
        prefix="brick_5",
    )
    return core


def linkin_modified() -> CoreModule:
    """Custom robot body built with the 3D editor."""
    core = CoreModule(index=0)
    hinge_0 = HingeModule(index=5)
    core.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_0.body,
        prefix="hinge_0",
    )
    hinge_1 = HingeModule(index=9)
    hinge_0.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_1.body,
        prefix="hinge_1",
    )
    hinge_2 = HingeModule(index=12)
    hinge_1.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_2.body,
        prefix="hinge_2",
    )
    hinge_3 = HingeModule(index=13)
    hinge_3.rotate(90)
    hinge_2.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_3.body,
        prefix="hinge_3",
    )
    brick_0 = BrickModule(index=14)
    hinge_3.sites[ModuleFaces.FRONT].attach_body(
        body=brick_0.body,
        prefix="brick_0",
    )
    brick_1 = BrickModule(index=15)
    brick_0.sites[ModuleFaces.FRONT].attach_body(
        body=brick_1.body,
        prefix="brick_1",
    )
    hinge_4 = HingeModule(index=20)
    brick_0.sites[ModuleFaces.BOTTOM].attach_body(
        body=hinge_4.body,
        prefix="hinge_4",
    )
    hinge_5 = HingeModule(index=21)
    hinge_4.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_5.body,
        prefix="hinge_5",
    )
    hinge_6 = HingeModule(index=24)
    hinge_6.rotate(90)
    brick_0.sites[ModuleFaces.TOP].attach_body(
        body=hinge_6.body,
        prefix="hinge_6",
    )
    hinge_7 = HingeModule(index=25)
    hinge_7.rotate(90)
    hinge_6.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_7.body,
        prefix="hinge_7",
    )
    hinge_8 = HingeModule(index=27)
    hinge_7.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_8.body,
        prefix="hinge_8",
    )
    hinge_9 = HingeModule(index=30)
    hinge_9.rotate(90)
    hinge_8.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_9.body,
        prefix="hinge_9",
    )
    hinge_10 = HingeModule(index=31)
    hinge_9.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_10.body,
        prefix="hinge_10",
    )
    hinge_11 = HingeModule(index=17)
    hinge_11.rotate(90)
    core.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_11.body,
        prefix="hinge_11",
    )
    brick_2 = BrickModule(index=18)
    hinge_11.sites[ModuleFaces.FRONT].attach_body(
        body=brick_2.body,
        prefix="brick_2",
    )
    return core


def snake() -> CoreModule:
    """Custom robot body built with the 3D editor."""
    core = CoreModule(index=0)
    hinge_0 = HingeModule(index=1)
    hinge_0.rotate(90)
    core.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_0.body,
        prefix="hinge_0",
    )
    brick_0 = BrickModule(index=2)
    hinge_0.sites[ModuleFaces.FRONT].attach_body(
        body=brick_0.body,
        prefix="brick_0",
    )
    hinge_1 = HingeModule(index=3)
    hinge_1.rotate(90)
    brick_0.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_1.body,
        prefix="hinge_1",
    )
    brick_1 = BrickModule(index=4)
    hinge_1.sites[ModuleFaces.FRONT].attach_body(
        body=brick_1.body,
        prefix="brick_1",
    )
    hinge_2 = HingeModule(index=5)
    hinge_2.rotate(90)
    brick_1.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_2.body,
        prefix="hinge_2",
    )
    brick_2 = BrickModule(index=6)
    hinge_2.sites[ModuleFaces.FRONT].attach_body(
        body=brick_2.body,
        prefix="brick_2",
    )
    hinge_3 = HingeModule(index=7)
    hinge_3.rotate(90)
    brick_2.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_3.body,
        prefix="hinge_3",
    )
    brick_3 = BrickModule(index=8)
    hinge_3.sites[ModuleFaces.FRONT].attach_body(
        body=brick_3.body,
        prefix="brick_3",
    )
    hinge_4 = HingeModule(index=9)
    hinge_4.rotate(90)
    brick_3.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_4.body,
        prefix="hinge_4",
    )
    brick_4 = BrickModule(index=10)
    hinge_4.sites[ModuleFaces.FRONT].attach_body(
        body=brick_4.body,
        prefix="brick_4",
    )
    hinge_5 = HingeModule(index=11)
    hinge_5.rotate(90)
    brick_4.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_5.body,
        prefix="hinge_5",
    )
    brick_5 = BrickModule(index=12)
    hinge_5.sites[ModuleFaces.FRONT].attach_body(
        body=brick_5.body,
        prefix="brick_5",
    )
    hinge_6 = HingeModule(index=13)
    hinge_6.rotate(90)
    brick_5.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_6.body,
        prefix="hinge_6",
    )
    brick_6 = BrickModule(index=14)
    hinge_6.sites[ModuleFaces.FRONT].attach_body(
        body=brick_6.body,
        prefix="brick_6",
    )
    hinge_7 = HingeModule(index=15)
    hinge_7.rotate(90)
    brick_6.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_7.body,
        prefix="hinge_7",
    )
    return core


def turtle() -> CoreModule:
    """Custom robot body built with the 3D editor."""
    core = CoreModule(index=0)
    brick_0 = BrickModule(index=1)
    core.sites[ModuleFaces.RIGHT].attach_body(
        body=brick_0.body,
        prefix="brick_0",
    )
    hinge_0 = HingeModule(index=2)
    hinge_0.rotate(90)
    brick_0.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_0.body,
        prefix="hinge_0",
    )
    hinge_1 = HingeModule(index=4)
    brick_0.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_1.body,
        prefix="hinge_1",
    )
    hinge_2 = HingeModule(index=6)
    hinge_2.rotate(90)
    hinge_1.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_2.body,
        prefix="hinge_2",
    )
    brick_1 = BrickModule(index=7)
    hinge_2.sites[ModuleFaces.FRONT].attach_body(
        body=brick_1.body,
        prefix="brick_1",
    )
    brick_2 = BrickModule(index=8)
    brick_1.sites[ModuleFaces.FRONT].attach_body(
        body=brick_2.body,
        prefix="brick_2",
    )
    hinge_3 = HingeModule(index=10)
    hinge_3.rotate(90)
    brick_2.sites[ModuleFaces.BOTTOM].attach_body(
        body=hinge_3.body,
        prefix="hinge_3",
    )
    hinge_4 = HingeModule(index=787)
    hinge_4.rotate(-180)
    brick_1.sites[ModuleFaces.TOP].attach_body(
        body=hinge_4.body,
        prefix="hinge_4",
    )
    brick_3 = BrickModule(index=788)
    hinge_4.sites[ModuleFaces.FRONT].attach_body(
        body=brick_3.body,
        prefix="brick_3",
    )
    brick_4 = BrickModule(index=789)
    brick_3.sites[ModuleFaces.FRONT].attach_body(
        body=brick_4.body,
        prefix="brick_4",
    )
    hinge_5 = HingeModule(index=790)
    hinge_5.rotate(90)
    brick_3.sites[ModuleFaces.TOP].attach_body(
        body=hinge_5.body,
        prefix="hinge_5",
    )
    hinge_6 = HingeModule(index=791)
    hinge_6.rotate(90)
    hinge_5.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_6.body,
        prefix="hinge_6",
    )
    hinge_7 = HingeModule(index=862)
    hinge_7.rotate(-180)
    brick_3.sites[ModuleFaces.BOTTOM].attach_body(
        body=hinge_7.body,
        prefix="hinge_7",
    )
    brick_5 = BrickModule(index=863)
    hinge_7.sites[ModuleFaces.FRONT].attach_body(
        body=brick_5.body,
        prefix="brick_5",
    )
    hinge_8 = HingeModule(index=864)
    brick_5.sites[ModuleFaces.BOTTOM].attach_body(
        body=hinge_8.body,
        prefix="hinge_8",
    )
    hinge_9 = HingeModule(index=865)
    hinge_9.rotate(90)
    brick_5.sites[ModuleFaces.TOP].attach_body(
        body=hinge_9.body,
        prefix="hinge_9",
    )
    hinge_10 = HingeModule(index=866)
    hinge_10.rotate(90)
    hinge_9.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_10.body,
        prefix="hinge_10",
    )
    hinge_11 = HingeModule(index=867)
    hinge_10.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_11.body,
        prefix="hinge_11",
    )
    hinge_12 = HingeModule(index=868)
    hinge_11.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_12.body,
        prefix="hinge_12",
    )
    return core


def iguana() -> CoreModule:
    """Custom robot body built with the 3D editor."""
    core = CoreModule(index=0)
    hinge_0 = HingeModule(index=897)
    hinge_0.rotate(-90)
    core.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_0.body,
        prefix="hinge_0",
    )
    hinge_1 = HingeModule(index=899)
    hinge_1.rotate(-90)
    hinge_0.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_1.body,
        prefix="hinge_1",
    )
    brick_0 = BrickModule(index=901)
    hinge_1.sites[ModuleFaces.FRONT].attach_body(
        body=brick_0.body,
        prefix="brick_0",
    )
    hinge_2 = HingeModule(index=898)
    hinge_2.rotate(-90)
    core.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_2.body,
        prefix="hinge_2",
    )
    hinge_3 = HingeModule(index=900)
    hinge_3.rotate(-90)
    hinge_2.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_3.body,
        prefix="hinge_3",
    )
    brick_1 = BrickModule(index=902)
    hinge_3.sites[ModuleFaces.FRONT].attach_body(
        body=brick_1.body,
        prefix="brick_1",
    )
    hinge_4 = HingeModule(index=904)
    core.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_4.body,
        prefix="hinge_4",
    )
    brick_2 = BrickModule(index=905)
    hinge_4.sites[ModuleFaces.FRONT].attach_body(
        body=brick_2.body,
        prefix="brick_2",
    )
    hinge_5 = HingeModule(index=906)
    brick_2.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_5.body,
        prefix="hinge_5",
    )
    brick_3 = BrickModule(index=907)
    hinge_5.sites[ModuleFaces.FRONT].attach_body(
        body=brick_3.body,
        prefix="brick_3",
    )
    hinge_6 = HingeModule(index=929)
    hinge_6.rotate(-45)
    brick_3.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_6.body,
        prefix="hinge_6",
    )
    brick_4 = BrickModule(index=975)
    hinge_6.sites[ModuleFaces.FRONT].attach_body(
        body=brick_4.body,
        prefix="brick_4",
    )
    hinge_7 = HingeModule(index=973)
    hinge_7.rotate(45)
    brick_3.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_7.body,
        prefix="hinge_7",
    )
    brick_5 = BrickModule(index=974)
    hinge_7.sites[ModuleFaces.FRONT].attach_body(
        body=brick_5.body,
        prefix="brick_5",
    )
    return core


def spider_8() -> CoreModule:
    """Custom robot body built with the 3D editor."""
    core = CoreModule(index=0)
    hinge_0 = HingeModule(index=2)
    core.sites[ModuleFaces.BACK].attach_body(
        body=hinge_0.body,
        prefix="hinge_0",
    )
    brick_0 = BrickModule(index=8)
    hinge_0.sites[ModuleFaces.FRONT].attach_body(
        body=brick_0.body,
        prefix="brick_0",
    )
    hinge_1 = HingeModule(index=11)
    hinge_1.rotate(90)
    brick_0.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_1.body,
        prefix="hinge_1",
    )
    brick_1 = BrickModule(index=18)
    hinge_1.sites[ModuleFaces.FRONT].attach_body(
        body=brick_1.body,
        prefix="brick_1",
    )
    hinge_2 = HingeModule(index=3)
    core.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_2.body,
        prefix="hinge_2",
    )
    brick_2 = BrickModule(index=9)
    hinge_2.sites[ModuleFaces.FRONT].attach_body(
        body=brick_2.body,
        prefix="brick_2",
    )
    hinge_3 = HingeModule(index=12)
    hinge_3.rotate(90)
    brick_2.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_3.body,
        prefix="hinge_3",
    )
    brick_3 = BrickModule(index=19)
    hinge_3.sites[ModuleFaces.FRONT].attach_body(
        body=brick_3.body,
        prefix="brick_3",
    )
    hinge_4 = HingeModule(index=4)
    core.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_4.body,
        prefix="hinge_4",
    )
    brick_4 = BrickModule(index=6)
    hinge_4.sites[ModuleFaces.FRONT].attach_body(
        body=brick_4.body,
        prefix="brick_4",
    )
    hinge_5 = HingeModule(index=26)
    hinge_5.rotate(90)
    brick_4.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_5.body,
        prefix="hinge_5",
    )
    brick_5 = BrickModule(index=27)
    hinge_5.sites[ModuleFaces.FRONT].attach_body(
        body=brick_5.body,
        prefix="brick_5",
    )
    hinge_6 = HingeModule(index=5)
    core.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_6.body,
        prefix="hinge_6",
    )
    brick_6 = BrickModule(index=7)
    hinge_6.sites[ModuleFaces.FRONT].attach_body(
        body=brick_6.body,
        prefix="brick_6",
    )
    hinge_7 = HingeModule(index=15)
    hinge_7.rotate(90)
    brick_6.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_7.body,
        prefix="hinge_7",
    )
    brick_7 = BrickModule(index=17)
    hinge_7.sites[ModuleFaces.FRONT].attach_body(
        body=brick_7.body,
        prefix="brick_7",
    )
    return core


def spider_12() -> CoreModule:
    """Custom robot body built with the 3D editor."""
    core = CoreModule(index=0)
    hinge_0 = HingeModule(index=2)
    core.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_0.body,
        prefix="hinge_0",
    )
    brick_0 = BrickModule(index=3)
    hinge_0.sites[ModuleFaces.FRONT].attach_body(
        body=brick_0.body,
        prefix="brick_0",
    )
    hinge_1 = HingeModule(index=10)
    hinge_1.rotate(90)
    brick_0.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_1.body,
        prefix="hinge_1",
    )
    brick_1 = BrickModule(index=11)
    hinge_1.sites[ModuleFaces.FRONT].attach_body(
        body=brick_1.body,
        prefix="brick_1",
    )
    hinge_2 = HingeModule(index=14)
    hinge_2.rotate(90)
    brick_1.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_2.body,
        prefix="hinge_2",
    )
    brick_2 = BrickModule(index=15)
    hinge_2.sites[ModuleFaces.FRONT].attach_body(
        body=brick_2.body,
        prefix="brick_2",
    )
    hinge_3 = HingeModule(index=8)
    core.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_3.body,
        prefix="hinge_3",
    )
    brick_3 = BrickModule(index=9)
    hinge_3.sites[ModuleFaces.FRONT].attach_body(
        body=brick_3.body,
        prefix="brick_3",
    )
    hinge_4 = HingeModule(index=16)
    hinge_4.rotate(90)
    brick_3.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_4.body,
        prefix="hinge_4",
    )
    brick_4 = BrickModule(index=17)
    hinge_4.sites[ModuleFaces.FRONT].attach_body(
        body=brick_4.body,
        prefix="brick_4",
    )
    hinge_5 = HingeModule(index=18)
    hinge_5.rotate(90)
    brick_4.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_5.body,
        prefix="hinge_5",
    )
    brick_5 = BrickModule(index=19)
    hinge_5.sites[ModuleFaces.FRONT].attach_body(
        body=brick_5.body,
        prefix="brick_5",
    )
    hinge_6 = HingeModule(index=22)
    core.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_6.body,
        prefix="hinge_6",
    )
    brick_6 = BrickModule(index=23)
    hinge_6.sites[ModuleFaces.FRONT].attach_body(
        body=brick_6.body,
        prefix="brick_6",
    )
    hinge_7 = HingeModule(index=29)
    hinge_7.rotate(90)
    brick_6.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_7.body,
        prefix="hinge_7",
    )
    brick_7 = BrickModule(index=30)
    hinge_7.sites[ModuleFaces.FRONT].attach_body(
        body=brick_7.body,
        prefix="brick_7",
    )
    hinge_8 = HingeModule(index=32)
    hinge_8.rotate(90)
    brick_7.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_8.body,
        prefix="hinge_8",
    )
    brick_8 = BrickModule(index=34)
    hinge_8.sites[ModuleFaces.FRONT].attach_body(
        body=brick_8.body,
        prefix="brick_8",
    )
    hinge_9 = HingeModule(index=24)
    core.sites[ModuleFaces.BACK].attach_body(
        body=hinge_9.body,
        prefix="hinge_9",
    )
    brick_9 = BrickModule(index=25)
    hinge_9.sites[ModuleFaces.FRONT].attach_body(
        body=brick_9.body,
        prefix="brick_9",
    )
    hinge_10 = HingeModule(index=28)
    hinge_10.rotate(90)
    brick_9.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_10.body,
        prefix="hinge_10",
    )
    brick_10 = BrickModule(index=31)
    hinge_10.sites[ModuleFaces.FRONT].attach_body(
        body=brick_10.body,
        prefix="brick_10",
    )
    hinge_11 = HingeModule(index=33)
    hinge_11.rotate(90)
    brick_10.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_11.body,
        prefix="hinge_11",
    )
    brick_11 = BrickModule(index=35)
    hinge_11.sites[ModuleFaces.FRONT].attach_body(
        body=brick_11.body,
        prefix="brick_11",
    )
    return core


def spider_16() -> CoreModule:
    """Custom robot body built with the 3D editor."""
    core = CoreModule(index=0)
    hinge_0 = HingeModule(index=2)
    core.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_0.body,
        prefix="hinge_0",
    )
    brick_0 = BrickModule(index=3)
    hinge_0.sites[ModuleFaces.FRONT].attach_body(
        body=brick_0.body,
        prefix="brick_0",
    )
    hinge_1 = HingeModule(index=10)
    hinge_1.rotate(90)
    brick_0.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_1.body,
        prefix="hinge_1",
    )
    brick_1 = BrickModule(index=11)
    hinge_1.sites[ModuleFaces.FRONT].attach_body(
        body=brick_1.body,
        prefix="brick_1",
    )
    hinge_2 = HingeModule(index=14)
    hinge_2.rotate(90)
    brick_1.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_2.body,
        prefix="hinge_2",
    )
    brick_2 = BrickModule(index=15)
    hinge_2.sites[ModuleFaces.FRONT].attach_body(
        body=brick_2.body,
        prefix="brick_2",
    )
    hinge_3 = HingeModule(index=37)
    hinge_3.rotate(90)
    brick_2.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_3.body,
        prefix="hinge_3",
    )
    brick_3 = BrickModule(index=44)
    hinge_3.sites[ModuleFaces.FRONT].attach_body(
        body=brick_3.body,
        prefix="brick_3",
    )
    hinge_4 = HingeModule(index=8)
    core.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_4.body,
        prefix="hinge_4",
    )
    brick_4 = BrickModule(index=9)
    hinge_4.sites[ModuleFaces.FRONT].attach_body(
        body=brick_4.body,
        prefix="brick_4",
    )
    hinge_5 = HingeModule(index=16)
    hinge_5.rotate(90)
    brick_4.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_5.body,
        prefix="hinge_5",
    )
    brick_5 = BrickModule(index=17)
    hinge_5.sites[ModuleFaces.FRONT].attach_body(
        body=brick_5.body,
        prefix="brick_5",
    )
    hinge_6 = HingeModule(index=18)
    hinge_6.rotate(90)
    brick_5.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_6.body,
        prefix="hinge_6",
    )
    brick_6 = BrickModule(index=19)
    hinge_6.sites[ModuleFaces.FRONT].attach_body(
        body=brick_6.body,
        prefix="brick_6",
    )
    hinge_7 = HingeModule(index=39)
    hinge_7.rotate(90)
    brick_6.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_7.body,
        prefix="hinge_7",
    )
    brick_7 = BrickModule(index=45)
    hinge_7.sites[ModuleFaces.FRONT].attach_body(
        body=brick_7.body,
        prefix="brick_7",
    )
    hinge_8 = HingeModule(index=22)
    core.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_8.body,
        prefix="hinge_8",
    )
    brick_8 = BrickModule(index=23)
    hinge_8.sites[ModuleFaces.FRONT].attach_body(
        body=brick_8.body,
        prefix="brick_8",
    )
    hinge_9 = HingeModule(index=29)
    hinge_9.rotate(90)
    brick_8.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_9.body,
        prefix="hinge_9",
    )
    brick_9 = BrickModule(index=30)
    hinge_9.sites[ModuleFaces.FRONT].attach_body(
        body=brick_9.body,
        prefix="brick_9",
    )
    hinge_10 = HingeModule(index=32)
    hinge_10.rotate(90)
    brick_9.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_10.body,
        prefix="hinge_10",
    )
    brick_10 = BrickModule(index=34)
    hinge_10.sites[ModuleFaces.FRONT].attach_body(
        body=brick_10.body,
        prefix="brick_10",
    )
    hinge_11 = HingeModule(index=40)
    hinge_11.rotate(90)
    brick_10.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_11.body,
        prefix="hinge_11",
    )
    brick_11 = BrickModule(index=42)
    hinge_11.sites[ModuleFaces.FRONT].attach_body(
        body=brick_11.body,
        prefix="brick_11",
    )
    hinge_12 = HingeModule(index=24)
    core.sites[ModuleFaces.BACK].attach_body(
        body=hinge_12.body,
        prefix="hinge_12",
    )
    brick_12 = BrickModule(index=25)
    hinge_12.sites[ModuleFaces.FRONT].attach_body(
        body=brick_12.body,
        prefix="brick_12",
    )
    hinge_13 = HingeModule(index=28)
    hinge_13.rotate(90)
    brick_12.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_13.body,
        prefix="hinge_13",
    )
    brick_13 = BrickModule(index=31)
    hinge_13.sites[ModuleFaces.FRONT].attach_body(
        body=brick_13.body,
        prefix="brick_13",
    )
    hinge_14 = HingeModule(index=33)
    hinge_14.rotate(90)
    brick_13.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_14.body,
        prefix="hinge_14",
    )
    brick_14 = BrickModule(index=35)
    hinge_14.sites[ModuleFaces.FRONT].attach_body(
        body=brick_14.body,
        prefix="brick_14",
    )
    hinge_15 = HingeModule(index=41)
    hinge_15.rotate(90)
    brick_14.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_15.body,
        prefix="hinge_15",
    )
    brick_15 = BrickModule(index=43)
    hinge_15.sites[ModuleFaces.FRONT].attach_body(
        body=brick_15.body,
        prefix="brick_15",
    )
    return core


def centipede_3() -> CoreModule:
    """Custom robot body built with the 3D editor."""
    core = CoreModule(index=0)
    hinge_0 = HingeModule(index=8)
    core.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_0.body,
        prefix="hinge_0",
    )
    brick_0 = BrickModule(index=9)
    hinge_0.sites[ModuleFaces.FRONT].attach_body(
        body=brick_0.body,
        prefix="brick_0",
    )
    hinge_1 = HingeModule(index=282)
    hinge_1.rotate(90)
    brick_0.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_1.body,
        prefix="hinge_1",
    )
    brick_1 = BrickModule(index=297)
    hinge_1.sites[ModuleFaces.FRONT].attach_body(
        body=brick_1.body,
        prefix="brick_1",
    )
    hinge_2 = HingeModule(index=283)
    hinge_2.rotate(90)
    brick_0.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_2.body,
        prefix="hinge_2",
    )
    brick_2 = BrickModule(index=298)
    hinge_2.sites[ModuleFaces.FRONT].attach_body(
        body=brick_2.body,
        prefix="brick_2",
    )
    hinge_3 = HingeModule(index=290)
    brick_0.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_3.body,
        prefix="hinge_3",
    )
    brick_3 = BrickModule(index=291)
    hinge_3.sites[ModuleFaces.FRONT].attach_body(
        body=brick_3.body,
        prefix="brick_3",
    )
    hinge_4 = HingeModule(index=299)
    brick_3.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_4.body,
        prefix="hinge_4",
    )
    brick_4 = BrickModule(index=300)
    hinge_4.sites[ModuleFaces.FRONT].attach_body(
        body=brick_4.body,
        prefix="brick_4",
    )
    hinge_5 = HingeModule(index=308)
    hinge_5.rotate(90)
    brick_4.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_5.body,
        prefix="hinge_5",
    )
    brick_5 = BrickModule(index=313)
    hinge_5.sites[ModuleFaces.FRONT].attach_body(
        body=brick_5.body,
        prefix="brick_5",
    )
    hinge_6 = HingeModule(index=309)
    hinge_6.rotate(90)
    brick_4.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_6.body,
        prefix="hinge_6",
    )
    brick_6 = BrickModule(index=316)
    hinge_6.sites[ModuleFaces.FRONT].attach_body(
        body=brick_6.body,
        prefix="brick_6",
    )
    hinge_7 = HingeModule(index=306)
    hinge_7.rotate(90)
    brick_3.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_7.body,
        prefix="hinge_7",
    )
    brick_7 = BrickModule(index=312)
    hinge_7.sites[ModuleFaces.FRONT].attach_body(
        body=brick_7.body,
        prefix="brick_7",
    )
    hinge_8 = HingeModule(index=307)
    hinge_8.rotate(90)
    brick_3.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_8.body,
        prefix="hinge_8",
    )
    brick_8 = BrickModule(index=315)
    hinge_8.sites[ModuleFaces.FRONT].attach_body(
        body=brick_8.body,
        prefix="brick_8",
    )
    hinge_9 = HingeModule(index=266)
    hinge_9.rotate(-90)
    core.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_9.body,
        prefix="hinge_9",
    )
    brick_9 = BrickModule(index=267)
    hinge_9.sites[ModuleFaces.FRONT].attach_body(
        body=brick_9.body,
        prefix="brick_9",
    )
    hinge_10 = HingeModule(index=278)
    hinge_10.rotate(90)
    core.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_10.body,
        prefix="hinge_10",
    )
    brick_10 = BrickModule(index=279)
    hinge_10.sites[ModuleFaces.FRONT].attach_body(
        body=brick_10.body,
        prefix="brick_10",
    )
    return core


def centipede_4() -> CoreModule:
    """Custom robot body built with the 3D editor."""
    core = CoreModule(index=0)
    hinge_0 = HingeModule(index=8)
    core.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_0.body,
        prefix="hinge_0",
    )
    brick_0 = BrickModule(index=9)
    hinge_0.sites[ModuleFaces.FRONT].attach_body(
        body=brick_0.body,
        prefix="brick_0",
    )
    hinge_1 = HingeModule(index=282)
    hinge_1.rotate(90)
    brick_0.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_1.body,
        prefix="hinge_1",
    )
    brick_1 = BrickModule(index=297)
    hinge_1.sites[ModuleFaces.FRONT].attach_body(
        body=brick_1.body,
        prefix="brick_1",
    )
    hinge_2 = HingeModule(index=283)
    hinge_2.rotate(90)
    brick_0.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_2.body,
        prefix="hinge_2",
    )
    brick_2 = BrickModule(index=298)
    hinge_2.sites[ModuleFaces.FRONT].attach_body(
        body=brick_2.body,
        prefix="brick_2",
    )
    hinge_3 = HingeModule(index=290)
    brick_0.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_3.body,
        prefix="hinge_3",
    )
    brick_3 = BrickModule(index=291)
    hinge_3.sites[ModuleFaces.FRONT].attach_body(
        body=brick_3.body,
        prefix="brick_3",
    )
    hinge_4 = HingeModule(index=299)
    brick_3.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_4.body,
        prefix="hinge_4",
    )
    brick_4 = BrickModule(index=300)
    hinge_4.sites[ModuleFaces.FRONT].attach_body(
        body=brick_4.body,
        prefix="brick_4",
    )
    hinge_5 = HingeModule(index=303)
    brick_4.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_5.body,
        prefix="hinge_5",
    )
    brick_5 = BrickModule(index=304)
    hinge_5.sites[ModuleFaces.FRONT].attach_body(
        body=brick_5.body,
        prefix="brick_5",
    )
    hinge_6 = HingeModule(index=310)
    hinge_6.rotate(90)
    brick_5.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_6.body,
        prefix="hinge_6",
    )
    brick_6 = BrickModule(index=314)
    hinge_6.sites[ModuleFaces.FRONT].attach_body(
        body=brick_6.body,
        prefix="brick_6",
    )
    hinge_7 = HingeModule(index=311)
    hinge_7.rotate(90)
    brick_5.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_7.body,
        prefix="hinge_7",
    )
    brick_7 = BrickModule(index=317)
    hinge_7.sites[ModuleFaces.FRONT].attach_body(
        body=brick_7.body,
        prefix="brick_7",
    )
    hinge_8 = HingeModule(index=308)
    hinge_8.rotate(90)
    brick_4.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_8.body,
        prefix="hinge_8",
    )
    brick_8 = BrickModule(index=313)
    hinge_8.sites[ModuleFaces.FRONT].attach_body(
        body=brick_8.body,
        prefix="brick_8",
    )
    hinge_9 = HingeModule(index=309)
    hinge_9.rotate(90)
    brick_4.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_9.body,
        prefix="hinge_9",
    )
    brick_9 = BrickModule(index=316)
    hinge_9.sites[ModuleFaces.FRONT].attach_body(
        body=brick_9.body,
        prefix="brick_9",
    )
    hinge_10 = HingeModule(index=306)
    hinge_10.rotate(90)
    brick_3.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_10.body,
        prefix="hinge_10",
    )
    brick_10 = BrickModule(index=312)
    hinge_10.sites[ModuleFaces.FRONT].attach_body(
        body=brick_10.body,
        prefix="brick_10",
    )
    hinge_11 = HingeModule(index=307)
    hinge_11.rotate(90)
    brick_3.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_11.body,
        prefix="hinge_11",
    )
    brick_11 = BrickModule(index=315)
    hinge_11.sites[ModuleFaces.FRONT].attach_body(
        body=brick_11.body,
        prefix="brick_11",
    )
    hinge_12 = HingeModule(index=266)
    hinge_12.rotate(-90)
    core.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_12.body,
        prefix="hinge_12",
    )
    brick_12 = BrickModule(index=267)
    hinge_12.sites[ModuleFaces.FRONT].attach_body(
        body=brick_12.body,
        prefix="brick_12",
    )
    hinge_13 = HingeModule(index=278)
    hinge_13.rotate(90)
    core.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_13.body,
        prefix="hinge_13",
    )
    brick_13 = BrickModule(index=279)
    hinge_13.sites[ModuleFaces.FRONT].attach_body(
        body=brick_13.body,
        prefix="brick_13",
    )
    return core


def centipede_5() -> CoreModule:
    """Custom robot body built with the 3D editor."""
    core = CoreModule(index=0)
    hinge_0 = HingeModule(index=8)
    core.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_0.body,
        prefix="hinge_0",
    )
    brick_0 = BrickModule(index=9)
    hinge_0.sites[ModuleFaces.FRONT].attach_body(
        body=brick_0.body,
        prefix="brick_0",
    )
    hinge_1 = HingeModule(index=282)
    hinge_1.rotate(90)
    brick_0.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_1.body,
        prefix="hinge_1",
    )
    brick_1 = BrickModule(index=297)
    hinge_1.sites[ModuleFaces.FRONT].attach_body(
        body=brick_1.body,
        prefix="brick_1",
    )
    hinge_2 = HingeModule(index=283)
    hinge_2.rotate(90)
    brick_0.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_2.body,
        prefix="hinge_2",
    )
    brick_2 = BrickModule(index=298)
    hinge_2.sites[ModuleFaces.FRONT].attach_body(
        body=brick_2.body,
        prefix="brick_2",
    )
    hinge_3 = HingeModule(index=290)
    brick_0.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_3.body,
        prefix="hinge_3",
    )
    brick_3 = BrickModule(index=291)
    hinge_3.sites[ModuleFaces.FRONT].attach_body(
        body=brick_3.body,
        prefix="brick_3",
    )
    hinge_4 = HingeModule(index=299)
    brick_3.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_4.body,
        prefix="hinge_4",
    )
    brick_4 = BrickModule(index=300)
    hinge_4.sites[ModuleFaces.FRONT].attach_body(
        body=brick_4.body,
        prefix="brick_4",
    )
    hinge_5 = HingeModule(index=303)
    brick_4.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_5.body,
        prefix="hinge_5",
    )
    brick_5 = BrickModule(index=304)
    hinge_5.sites[ModuleFaces.FRONT].attach_body(
        body=brick_5.body,
        prefix="brick_5",
    )
    hinge_6 = HingeModule(index=310)
    hinge_6.rotate(90)
    brick_5.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_6.body,
        prefix="hinge_6",
    )
    brick_6 = BrickModule(index=314)
    hinge_6.sites[ModuleFaces.FRONT].attach_body(
        body=brick_6.body,
        prefix="brick_6",
    )
    hinge_7 = HingeModule(index=311)
    hinge_7.rotate(90)
    brick_5.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_7.body,
        prefix="hinge_7",
    )
    brick_7 = BrickModule(index=317)
    hinge_7.sites[ModuleFaces.FRONT].attach_body(
        body=brick_7.body,
        prefix="brick_7",
    )
    hinge_8 = HingeModule(index=319)
    brick_5.sites[ModuleFaces.FRONT].attach_body(
        body=hinge_8.body,
        prefix="hinge_8",
    )
    brick_8 = BrickModule(index=320)
    hinge_8.sites[ModuleFaces.FRONT].attach_body(
        body=brick_8.body,
        prefix="brick_8",
    )
    hinge_9 = HingeModule(index=323)
    hinge_9.rotate(90)
    brick_8.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_9.body,
        prefix="hinge_9",
    )
    brick_9 = BrickModule(index=325)
    hinge_9.sites[ModuleFaces.FRONT].attach_body(
        body=brick_9.body,
        prefix="brick_9",
    )
    hinge_10 = HingeModule(index=324)
    hinge_10.rotate(90)
    brick_8.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_10.body,
        prefix="hinge_10",
    )
    brick_10 = BrickModule(index=326)
    hinge_10.sites[ModuleFaces.FRONT].attach_body(
        body=brick_10.body,
        prefix="brick_10",
    )
    hinge_11 = HingeModule(index=308)
    hinge_11.rotate(90)
    brick_4.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_11.body,
        prefix="hinge_11",
    )
    brick_11 = BrickModule(index=313)
    hinge_11.sites[ModuleFaces.FRONT].attach_body(
        body=brick_11.body,
        prefix="brick_11",
    )
    hinge_12 = HingeModule(index=309)
    hinge_12.rotate(90)
    brick_4.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_12.body,
        prefix="hinge_12",
    )
    brick_12 = BrickModule(index=316)
    hinge_12.sites[ModuleFaces.FRONT].attach_body(
        body=brick_12.body,
        prefix="brick_12",
    )
    hinge_13 = HingeModule(index=306)
    hinge_13.rotate(90)
    brick_3.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_13.body,
        prefix="hinge_13",
    )
    brick_13 = BrickModule(index=312)
    hinge_13.sites[ModuleFaces.FRONT].attach_body(
        body=brick_13.body,
        prefix="brick_13",
    )
    hinge_14 = HingeModule(index=307)
    hinge_14.rotate(90)
    brick_3.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_14.body,
        prefix="hinge_14",
    )
    brick_14 = BrickModule(index=315)
    hinge_14.sites[ModuleFaces.FRONT].attach_body(
        body=brick_14.body,
        prefix="brick_14",
    )
    hinge_15 = HingeModule(index=266)
    hinge_15.rotate(-90)
    core.sites[ModuleFaces.RIGHT].attach_body(
        body=hinge_15.body,
        prefix="hinge_15",
    )
    brick_15 = BrickModule(index=267)
    hinge_15.sites[ModuleFaces.FRONT].attach_body(
        body=brick_15.body,
        prefix="brick_15",
    )
    hinge_16 = HingeModule(index=278)
    hinge_16.rotate(90)
    core.sites[ModuleFaces.LEFT].attach_body(
        body=hinge_16.body,
        prefix="hinge_16",
    )
    brick_16 = BrickModule(index=279)
    hinge_16.sites[ModuleFaces.FRONT].attach_body(
        body=brick_16.body,
        prefix="brick_16",
    )
    return core