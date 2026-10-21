"""Tests for variable-length BrickModule geometry and mass scaling."""

import mujoco
import pytest

from ariel.body_phenotypes.robogen_lite.modules.brick import (
    BrickModule,
    mass_from_length,
)
from ariel.parameters.ariel_modules import ArielModulesConfig


config = ArielModulesConfig()


def test_default_brick_length_matches_config() -> None:
    """A BrickModule without an explicit length uses the configured default."""
    brick = BrickModule(index=1)

    assert brick.length == pytest.approx(
        config.BRICK_LENGTH_DEFAULT
    )


@pytest.mark.parametrize(
    ("length", "expected_mass"),
    [
        (
            0.075,
            0.04265,
        ),
        (
            0.150,
            0.04268339857142825,
        ),
        (
            0.225,
            0.0427167971428565,
        ),
    ],
)
def test_mass_from_length_uses_expected_scaling(
    length: float,
    expected_mass: float,
) -> None:
    """Mass should follow the configured variable-length scaling.
    
    The public BrickModule API uses meters, while the source linear mass slope
    is expressed per millimetre. This test intentionally catches the common
    unit-conversion bug where the metre delta is multiplied directly by the
    g/mm coefficient.
    """
    assert mass_from_length(length) == pytest.approx(
        expected_mass,
        rel=1e-9,
        abs=1e-12,
    )


def test_mass_increases_with_length() -> None:
    """Longer bricks must be heavier than shorter bricks."""
    short = mass_from_length(
        config.BRICK_LENGTH_MIN
    )
    long = mass_from_length(
        config.BRICK_LENGTH_MAX
    )

    assert long > short


@pytest.mark.parametrize(
    "length",
    [
        0.075,
        0.150,
        0.225,
    ],
)
def test_brick_compiles_at_supported_lengths(
    length: float,
) -> None:
    """Representative supported brick lengths should compile in MuJoCo."""
    brick = BrickModule(
        index=1,
        length=length,
    )

    model = brick.spec.compile()

    assert isinstance(
        model,
        mujoco.MjModel,
    )


@pytest.mark.parametrize(
    "length",
    [
        0.074999,
        0.0,
        -0.1,
        0.225001,
        0.5,
    ],
)
def test_brick_rejects_out_of_range_lengths(
    length: float,
) -> None:
    """Brick lengths outside the configured interval are invalid."""
    with pytest.raises(
        ValueError
    ):
        BrickModule(
            index=1,
            length=length,
        )


def test_brick_geom_length_matches_requested_length() -> None:
    """The MuJoCo box must have the requested full physical length."""
    requested_length = 0.180

    brick = BrickModule(
        index=1,
        length=requested_length,
    )
    model = brick.spec.compile()

    # There is one brick geom in this standalone module.
    assert model.ngeom >= 1

    # MuJoCo box size stores half extents.
    half_extent_y = float(
        model.geom_size[
            0,
            1,
        ]
    )

    assert (
        2.0 * half_extent_y
        == pytest.approx(
            requested_length
        )
    )


def test_brick_front_site_moves_with_length() -> None:
    """The front attachment site should remain at the end of the brick."""
    requested_length = 0.180

    brick = BrickModule(
        index=1,
        length=requested_length,
    )
    model = brick.spec.compile()

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "brick-front",
    )

    assert site_id >= 0

    assert float(
        model.site_pos[
            site_id,
            1,
        ]
    ) == pytest.approx(
        requested_length
    )
