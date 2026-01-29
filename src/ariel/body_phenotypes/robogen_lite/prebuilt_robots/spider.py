"""TODO(A-lamo): description of script."""

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule


def spider() -> CoreModule:
    """Spider robot body.
    
    Body Description
    ---------
    The spider body consists of a core module, 4 legs. Each leg constains two hinges,
    with blocks attached at the end of them.

    Returns
    -------
    CoreModule
        The core module of the robot, with all other modules attached.
    """
    core = CoreModule(
        index=0,
    )

    # --- LEG 1 (Front) ---
    l1_1 = HingeModule(
        index=1,
    )
    l1_1.rotate(0)  # Horizontal (Yaw)

    l1_2 = HingeModule(
        index=2,
    )
    l1_2.rotate(90) # Vertical (Pitch)

    l1_3 = BrickModule(
        index=3,
    )

    # --- LEG 2 (Left) ---
    l2_1 = HingeModule(
        index=4,
    )
    l2_1.rotate(0)  # Horizontal (Yaw)

    l2_2 = HingeModule(
        index=5,
    )
    l2_2.rotate(90) # Vertical (Pitch) 

    l2_3 = BrickModule(
        index=6,
    )

    # --- LEG 3 (Right) ---
    l3_1 = HingeModule(
        index=7,
    )
    l3_1.rotate(0)  # Horizontal (Yaw)

    l3_2 = HingeModule(
        index=8,
    )
    l3_2.rotate(90) # Vertical (Pitch)

    l3_3 = BrickModule(
        index=9,
    )

    # --- LEG 4 (Back) ---
    l4_1 = HingeModule(
        index=10,
    )
    l4_1.rotate(0)  # Horizontal (Yaw)

    l4_2 = HingeModule(
        index=11,
    )
    l4_2.rotate(90) # Vertical (Pitch)

    l4_3 = BrickModule(
        index=12,
    )

    # --- ATTACH BODIES ---

    """Front-leg"""
    core.sites[ModuleFaces.FRONT].attach_body(
        body=l1_1.body,
        prefix="l1_1",
    )
    l1_1.sites[ModuleFaces.FRONT].attach_body(
        body=l1_2.body,
        prefix="l1_2",
    )
    l1_2.sites[ModuleFaces.FRONT].attach_body(
        body=l1_3.body,
        prefix="l1_3",
    )

    """Left-leg"""
    core.sites[ModuleFaces.LEFT].attach_body(
        body=l2_1.body,
        prefix="l2_1",
    )
    l2_1.sites[ModuleFaces.FRONT].attach_body(
        body=l2_2.body,
        prefix="l2_2",
    )
    l2_2.sites[ModuleFaces.FRONT].attach_body(
        body=l2_3.body,
        prefix="l2_3",
    )

    """Right-leg"""
    core.sites[ModuleFaces.RIGHT].attach_body(
        body=l3_1.body,
        prefix="l3_1",
    )
    l3_1.sites[ModuleFaces.FRONT].attach_body(
        body=l3_2.body,
        prefix="l3_2",
    )
    l3_2.sites[ModuleFaces.FRONT].attach_body(
        body=l3_3.body,
        prefix="l3_3",
    )

    """Back-leg"""
    core.sites[ModuleFaces.BACK].attach_body(
        body=l4_1.body,
        prefix="l4_1",
    )
    l4_1.sites[ModuleFaces.FRONT].attach_body(
        body=l4_2.body,
        prefix="l4_2",
    )
    l4_2.sites[ModuleFaces.FRONT].attach_body(
        body=l4_3.body,
        prefix="l4_3",
    )

    return core