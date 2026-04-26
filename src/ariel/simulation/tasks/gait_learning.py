"""Gait learning."""


def xy_displacement(
    xy1: tuple[float, float],
    xy2: tuple[float, float],
) -> float:
    """
    Calculate the displacement between two points in 2D space.

    Parameters
    ----------
    xy1
        Coordinates of the first point (x1, y1).
    xy2
        Coordinates of the second point (x2, y2).

    Returns
    -------
    float
        The Euclidean distance between the two points.
    """
    return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5


def x_speed(
    xy1: tuple[float, float],
    xy2: tuple[float, float],
    dt: float,
) -> float:
    """
    Calculate the speed in the x direction between two points.

    Parameters
    ----------
    xy1
        Coordinates of the first point (x1, y1).
    xy2
        Coordinates of the second point (x2, y2).
    dt
        Time difference between the two points.

    Returns
    -------
    float
        The speed in the x direction.
    """
    return abs(xy2[0] - xy1[0]) / dt if dt > 0 else 0.0


def y_speed(
    xy1: tuple[float, float],
    xy2: tuple[float, float],
    dt: float,
) -> float:
    """
    Calculate the speed in the y direction between two points.

    Parameters
    ----------
    xy1
        Coordinates of the first point (x1, y1).
    xy2
        Coordinates of the second point (x2, y2).
    dt
        Time difference between the two points.

    Returns
    -------
    float
        The speed in the y direction.
    """
    return abs(xy2[1] - xy1[1]) / dt if dt > 0 else 0.0
