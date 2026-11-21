"""Tests for CPPN variable-length brick decoding helpers."""

import pytest

from ariel.body_phenotypes.robogen_lite.decoders.cppn_best_first import (
    scale_brick_length,
)
from ariel.parameters.ariel_modules import ArielModulesConfig


config = ArielModulesConfig()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0.0, 0.075),
        (0.25, 0.1125),
        (0.5, 0.150),
        (0.75, 0.1875),
        (1.0, 0.225),
    ],
)
def test_scale_brick_length(
    raw: float,
    expected: float,
) -> None:
    """CPPN length output should map linearly into the valid range."""
    assert scale_brick_length(
        raw
    ) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (-100.0, 0.075),
        (-1.0, 0.075),
        (2.0, 0.225),
        (100.0, 0.225),
    ],
)
def test_scale_brick_length_clamps(
    raw: float,
    expected: float,
) -> None:
    """Out-of-range CPPN outputs should be clipped safely."""
    assert scale_brick_length(
        raw
    ) == pytest.approx(
        expected
    )


def test_scale_endpoints_match_configuration() -> None:
    """Decoder scaling must use the configured limits."""
    assert scale_brick_length(
        0.0
    ) == pytest.approx(
        config.BRICK_LENGTH_MIN
    )

    assert scale_brick_length(
        1.0
    ) == pytest.approx(
        config.BRICK_LENGTH_MAX
    )
