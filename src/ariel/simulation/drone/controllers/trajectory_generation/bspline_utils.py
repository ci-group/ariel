# -*- coding: utf-8 -*-
"""
B-Spline Utilities for Trajectory Generation

This module provides core B-spline mathematical functions for generating
smooth trajectories. Implements cubic B-splines (degree 3) with support
for clamped and periodic boundary conditions.

Author: Generated for AirEvolve project
License: MIT
"""

import numpy as np
from typing import Tuple, Optional


def generate_knot_vector(n_control_points: int, degree: int,
                         boundary: str = 'clamped') -> np.ndarray:
    """
    Generate knot vector for B-spline curve.

    Args:
        n_control_points: Number of control points
        degree: Degree of B-spline (use 3 for cubic)
        boundary: Boundary condition - 'clamped' or 'periodic'
                 'clamped': curve interpolates first and last control points
                 'periodic': curve forms a closed loop

    Returns:
        Knot vector of length (n_control_points + degree + 1)

    Example:
        For 5 control points, degree 3, clamped:
        knots = [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]
        (4 repeated at start, 4 repeated at end)
    """
    if boundary == 'clamped':
        # Clamped B-spline: repeat first and last knots (degree+1) times
        # Internal knots are uniform
        n_internal = n_control_points - degree
        if n_internal < 1:
            raise ValueError(f"Need at least {degree+1} control points for degree {degree}")

        # Create uniform internal knots strictly between 0 and max_knot
        # For n CPs and degree p, we need n-p-1 distinct internal knots
        if n_internal == 1:
            # Minimal case: only boundary knots
            internal_knots = np.array([])
            max_knot = 1.0
        else:
            internal_knots = np.arange(1, n_internal)
            max_knot = n_internal

        # Add repeated boundary knots
        knots = np.concatenate([
            np.zeros(degree + 1),       # Repeated start knots (degree+1 times)
            internal_knots,             # Internal knots
            np.full(degree + 1, max_knot if n_internal > 1 else 1.0)  # Repeated end knots
        ])

    elif boundary == 'periodic':
        # Periodic B-spline: uniform knots for closed loop
        # Need n + 2*degree + 1 knots for proper periodic spline
        n_knots = n_control_points + 2 * degree + 1
        knots = np.arange(n_knots) - degree

    else:
        raise ValueError(f"Unknown boundary condition: {boundary}")

    return knots


def basis_function(i: int, p: int, u: float, knots: np.ndarray) -> float:
    """
    Compute B-spline basis function N_{i,p}(u) using Cox-de Boor recursion.

    Args:
        i: Index of basis function
        p: Degree of basis function
        u: Parameter value
        knots: Knot vector

    Returns:
        Value of basis function at u

    Note:
        This implements the Cox-de Boor recursion formula:
        N_{i,0}(u) = 1 if knots[i] <= u < knots[i+1], else 0
        N_{i,p}(u) = ((u - knots[i]) / (knots[i+p] - knots[i])) * N_{i,p-1}(u)
                   + ((knots[i+p+1] - u) / (knots[i+p+1] - knots[i+1])) * N_{i+1,p-1}(u)
    """
    # Base case: degree 0
    if p == 0:
        return 1.0 if knots[i] <= u < knots[i + 1] else 0.0

    # Recursive case
    # Handle division by zero (when knot span is zero)
    denom1 = knots[i + p] - knots[i]
    term1 = ((u - knots[i]) / denom1) * basis_function(i, p - 1, u, knots) if denom1 != 0 else 0.0

    denom2 = knots[i + p + 1] - knots[i + 1]
    term2 = ((knots[i + p + 1] - u) / denom2) * basis_function(i + 1, p - 1, u, knots) if denom2 != 0 else 0.0

    return term1 + term2


def basis_function_derivative(i: int, p: int, u: float, knots: np.ndarray,
                               deriv_order: int = 1) -> float:
    """
    Compute derivative of B-spline basis function.

    Args:
        i: Index of basis function
        p: Degree of basis function
        u: Parameter value
        knots: Knot vector
        deriv_order: Order of derivative (1 for first derivative, 2 for second)

    Returns:
        Value of basis function derivative at u

    Note:
        First derivative: N'_{i,p}(u) = p * (N_{i,p-1}(u) / (knots[i+p] - knots[i])
                                             - N_{i+1,p-1}(u) / (knots[i+p+1] - knots[i+1]))
    """
    if deriv_order == 0:
        return basis_function(i, p, u, knots)

    if p == 0:
        return 0.0

    # First derivative
    denom1 = knots[i + p] - knots[i]
    term1 = basis_function_derivative(i, p - 1, u, knots, deriv_order - 1) / denom1 if denom1 != 0 else 0.0

    denom2 = knots[i + p + 1] - knots[i + 1]
    term2 = basis_function_derivative(i + 1, p - 1, u, knots, deriv_order - 1) / denom2 if denom2 != 0 else 0.0

    return p * (term1 - term2)


def evaluate_bspline(u: float, control_points: np.ndarray, knots: np.ndarray,
                     degree: int = 3) -> np.ndarray:
    """
    Evaluate B-spline curve at parameter u.

    Args:
        u: Parameter value (typically in range [0, max_u])
        control_points: Array of control points, shape (n, d) where d is dimension (usually 3 for xyz)
        knots: Knot vector
        degree: Degree of B-spline (default 3 for cubic)

    Returns:
        Point on curve at parameter u, shape (d,)

    Example:
        >>> control_points = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0]])
        >>> knots = generate_knot_vector(4, 3, 'clamped')
        >>> point = evaluate_bspline(0.5, control_points, knots, 3)
    """
    n = len(control_points)

    # Clamp u to valid range
    u = np.clip(u, knots[degree], knots[n])

    # Handle edge case at the end
    if u == knots[n]:
        u = knots[n] - 1e-10

    # Find which knot span u is in
    span = np.searchsorted(knots, u, side='right') - 1
    span = max(degree, min(span, n - 1))

    # Compute point using basis functions
    point = np.zeros(control_points.shape[1])
    for i in range(n):
        basis = basis_function(i, degree, u, knots)
        point += basis * control_points[i]

    return point


def evaluate_bspline_derivative(u: float, control_points: np.ndarray,
                                knots: np.ndarray, degree: int = 3,
                                deriv_order: int = 1) -> np.ndarray:
    """
    Evaluate derivative of B-spline curve at parameter u.

    Args:
        u: Parameter value
        control_points: Array of control points, shape (n, d)
        knots: Knot vector
        degree: Degree of B-spline
        deriv_order: Order of derivative (1 for velocity, 2 for acceleration)

    Returns:
        Derivative vector at parameter u, shape (d,)

    Note:
        For first derivative (velocity), this returns dC(u)/du
        For second derivative (acceleration), this returns d²C(u)/du²
        These need to be scaled by du/dt to get physical velocity/acceleration
    """
    n = len(control_points)

    # Clamp u to valid range
    u = np.clip(u, knots[degree], knots[n])

    # Handle edge case at the end
    if u == knots[n]:
        u = knots[n] - 1e-10

    # Compute derivative using basis function derivatives
    derivative = np.zeros(control_points.shape[1])
    for i in range(n):
        basis_deriv = basis_function_derivative(i, degree, u, knots, deriv_order)
        derivative += basis_deriv * control_points[i]

    return derivative


class BSplineCurve:
    """
    B-spline curve class for convenient evaluation.

    This class wraps the B-spline functions and provides methods for
    evaluating position, velocity, and acceleration along the curve.
    """

    def __init__(self, control_points: np.ndarray, degree: int = 3,
                 boundary: str = 'clamped'):
        """
        Initialize B-spline curve.

        Args:
            control_points: Array of control points, shape (n, d)
            degree: Degree of B-spline (default 3 for cubic)
            boundary: Boundary condition - 'clamped' or 'periodic'
        """
        self.control_points = np.array(control_points)
        self.degree = degree
        self.boundary = boundary
        self.n_points = len(control_points)

        # Generate knot vector
        self.knots = generate_knot_vector(self.n_points, degree, boundary)

        # Parameter range
        self.u_min = self.knots[degree]
        if boundary == 'periodic':
            # For periodic splines, one full period spans n_points parameter units
            self.u_max = self.u_min + self.n_points
        else:
            # For clamped splines, use the last knot
            self.u_max = self.knots[self.n_points]

    def __call__(self, u: float) -> np.ndarray:
        """Evaluate curve at parameter u (position)."""
        return evaluate_bspline(u, self.control_points, self.knots, self.degree)

    def position(self, u: float) -> np.ndarray:
        """Evaluate position at parameter u."""
        # For periodic splines, handle wrapping properly
        if self.boundary == 'periodic':
            return self._evaluate_periodic(u)
        else:
            return evaluate_bspline(u, self.control_points, self.knots, self.degree)

    def velocity(self, u: float, du_dt: float = 1.0) -> np.ndarray:
        """
        Evaluate velocity at parameter u.

        Args:
            u: Parameter value
            du_dt: Time derivative of parameter (du/dt)

        Returns:
            Velocity vector = dC/du * du/dt
        """
        if self.boundary == 'periodic':
            deriv = self._evaluate_periodic_derivative(u, deriv_order=1)
        else:
            deriv = evaluate_bspline_derivative(u, self.control_points, self.knots,
                                               self.degree, deriv_order=1)
        return deriv * du_dt

    def acceleration(self, u: float, du_dt: float = 1.0, d2u_dt2: float = 0.0) -> np.ndarray:
        """
        Evaluate acceleration at parameter u.

        Args:
            u: Parameter value
            du_dt: First time derivative of parameter (du/dt)
            d2u_dt2: Second time derivative of parameter (d²u/dt²)

        Returns:
            Acceleration vector = d²C/du² * (du/dt)² + dC/du * d²u/dt²
        """
        if self.boundary == 'periodic':
            first_deriv = self._evaluate_periodic_derivative(u, deriv_order=1)
            second_deriv = self._evaluate_periodic_derivative(u, deriv_order=2)
        else:
            first_deriv = evaluate_bspline_derivative(u, self.control_points, self.knots,
                                                      self.degree, deriv_order=1)
            second_deriv = evaluate_bspline_derivative(u, self.control_points, self.knots,
                                                       self.degree, deriv_order=2)

        return second_deriv * (du_dt ** 2) + first_deriv * d2u_dt2

    def _evaluate_periodic(self, u: float) -> np.ndarray:
        """
        Evaluate periodic B-spline at parameter u with proper control point wrapping.

        For periodic splines, control points wrap: P_i = P_{i mod n}
        """
        # Clamp to valid range
        u = np.clip(u, self.u_min, self.u_max - 1e-10)

        # For periodic splines, we need to sum over all basis functions that might be non-zero
        # At parameter u, basis functions N_{i,p}(u) for i in a certain range are non-zero
        # We need to check basis functions that might reference our control points (with wrapping)
        point = np.zeros(self.control_points.shape[1])

        # For degree p, at most p+1 basis functions are non-zero at any u
        # Find the knot span and check basis functions around it
        # Check all basis functions and use wrapped control point indices
        for i in range(len(self.knots) - self.degree - 1):
            basis = basis_function(i, self.degree, u, self.knots)
            if abs(basis) > 1e-10:  # Only accumulate if non-zero
                # Wrap control point index
                cp_idx = i % self.n_points
                point += basis * self.control_points[cp_idx]

        return point

    def _evaluate_periodic_derivative(self, u: float, deriv_order: int = 1) -> np.ndarray:
        """
        Evaluate derivative of periodic B-spline at parameter u with proper control point wrapping.
        """
        # Clamp to valid range
        u = np.clip(u, self.u_min, self.u_max - 1e-10)

        # Compute derivative using basis function derivatives with wrapped control points
        derivative = np.zeros(self.control_points.shape[1])

        for i in range(len(self.knots) - self.degree - 1):
            basis_deriv = basis_function_derivative(i, self.degree, u, self.knots, deriv_order)
            if abs(basis_deriv) > 1e-10:  # Only accumulate if non-zero
                # Wrap control point index
                cp_idx = i % self.n_points
                derivative += basis_deriv * self.control_points[cp_idx]

        return derivative

    def sample_curve(self, n_samples: int = 100) -> np.ndarray:
        """
        Sample the curve at uniformly spaced parameter values.

        Args:
            n_samples: Number of samples

        Returns:
            Array of points on curve, shape (n_samples, d)
        """
        u_values = np.linspace(self.u_min, self.u_max, n_samples)
        return np.array([self.position(u) for u in u_values])


def uniform_time_parameterization(spline: BSplineCurve, total_time: float,
                                  velocity_scale: float = 1.0) -> Tuple[callable, callable, callable]:
    """
    Create time-parameterized functions for a B-spline curve.

    This function creates a uniform time parameterization where the parameter u
    varies linearly with time: u(t) = (u_max - u_min) * (t / total_time) + u_min

    Args:
        spline: BSplineCurve object
        total_time: Total time to traverse the curve
        velocity_scale: Scale factor for velocity (>1 = faster, <1 = slower)

    Returns:
        Tuple of (position_func, velocity_func, acceleration_func)
        Each function takes time t and returns the corresponding vector

    Example:
        >>> pos_func, vel_func, acc_func = uniform_time_parameterization(spline, 10.0)
        >>> position_at_5s = pos_func(5.0)
    """
    u_range = spline.u_max - spline.u_min

    # du/dt is constant for uniform parameterization
    du_dt = (u_range / total_time) * velocity_scale

    def position_func(t: float) -> np.ndarray:
        """Get position at time t."""
        u = spline.u_min + (u_range * t / total_time)
        u = np.clip(u, spline.u_min, spline.u_max)
        return spline.position(u)

    def velocity_func(t: float) -> np.ndarray:
        """Get velocity at time t."""
        u = spline.u_min + (u_range * t / total_time)
        u = np.clip(u, spline.u_min, spline.u_max)
        return spline.velocity(u, du_dt)

    def acceleration_func(t: float) -> np.ndarray:
        """Get acceleration at time t."""
        u = spline.u_min + (u_range * t / total_time)
        u = np.clip(u, spline.u_min, spline.u_max)
        # d²u/dt² = 0 for uniform parameterization
        return spline.acceleration(u, du_dt, 0.0)

    return position_func, velocity_func, acceleration_func
