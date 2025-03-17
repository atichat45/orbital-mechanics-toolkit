"""
Lambert's Problem Solver

This module provides functions for solving Lambert's problem, which finds the orbit
connecting two points in space with a specified time of flight.
"""

import numpy as np
from numpy.linalg import norm

def solve_lambert(r1, r2, tof, mu, clockwise=False, max_iter=100, tol=1e-10):
    """
    Solve Lambert's problem using Battin's method.
    
    Args:
        r1: Initial position vector [x, y, z] (m)
        r2: Final position vector [x, y, z] (m)
        tof: Time of flight (s)
        mu: Gravitational parameter (m^3/s^2)
        clockwise: If True, choose clockwise motion (default: False)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
    
    Returns:
        Tuple of initial and final velocity vectors [v1, v2]
    """
    r1 = np.array(r1)
    r2 = np.array(r2)
    
    # Magnitudes
    r1_norm = norm(r1)
    r2_norm = norm(r2)
    
    # Cross product of position vectors to determine orbital plane
    r1xr2 = np.cross(r1, r2)
    
    # If r1xr2 is zero, the points and the center are collinear
    if norm(r1xr2) < tol:
        # Handle the collinear case with a small perturbation
        r2 = r2 + np.array([1e-10, 1e-10, 1e-10])
        r1xr2 = np.cross(r1, r2)
    
    # Check if clockwise motion is requested
    if clockwise:
        r1xr2 = -r1xr2
    
    # Unit vector normal to the transfer plane
    h_hat = r1xr2 / norm(r1xr2)
    
    # Angle between position vectors (0 to pi)
    cos_dnu = np.dot(r1, r2) / (r1_norm * r2_norm)
    cos_dnu = np.clip(cos_dnu, -1.0, 1.0)  # Ensure domain -1 <= cos <= 1
    dnu = np.arccos(cos_dnu)
    
    # Check if transfer angle exceeds pi (long way around)
    cross_product = np.cross(r1, r2)
    if np.dot(r1xr2, cross_product) < 0:
        dnu = 2 * np.pi - dnu
    
    # Semiperimeter and chord
    c = np.sqrt(r1_norm**2 + r2_norm**2 - 2 * r1_norm * r2_norm * np.cos(dnu))
    s = (r1_norm + r2_norm + c) / 2
    
    # Minimum energy ellipse semi-major axis (parabolic limit)
    a_min = s / 2
    
    # Time of flight for minimum energy ellipse
    alpha = 2 * np.arcsin(np.sqrt(s / (2 * a_min)))
    beta = 2 * np.arcsin(np.sqrt((s - c) / (2 * a_min)))
    if dnu > np.pi:
        beta = -beta
    
    t_min = np.sqrt(a_min**3 / mu) * (alpha - beta - np.sin(alpha) + np.sin(beta))
    
    if tof <= t_min + tol:
        # For parabolic and near-parabolic transfers
        a = (r1_norm + r2_norm + c) / 4
        
        # Construct velocity vectors for parabolic orbit
        f = 1 - r2_norm / r1_norm
        g = r1_norm * r2_norm * np.sin(dnu) / np.sqrt(mu * a_min)
        g_dot = 1 - r1_norm / r2_norm
        
        v1 = (r2 - f * r1) / g
        v2 = (g_dot * r2 - r1) / g
        
        return v1, v2
    
    # For elliptical transfers (most common case)
    # Initialize with a guess slightly larger than a_min
    a = a_min * 1.05
    
    # Solve for the semi-major axis using iterative approach
    for i in range(max_iter):
        alpha = 2 * np.arcsin(np.sqrt(s / (2 * a)))
        beta = 2 * np.arcsin(np.sqrt((s - c) / (2 * a)))
        
        if dnu > np.pi:
            beta = -beta
        
        # Calculate time of flight for current guess
        t_a = np.sqrt(a**3 / mu) * (alpha - beta - np.sin(alpha) + np.sin(beta))
        
        # Check convergence
        if abs(t_a - tof) < tol:
            break
        
        # Numerical differentiation for derivative
        a_plus = a * 1.01
        alpha_plus = 2 * np.arcsin(np.sqrt(s / (2 * a_plus)))
        beta_plus = 2 * np.arcsin(np.sqrt((s - c) / (2 * a_plus)))
        
        if dnu > np.pi:
            beta_plus = -beta_plus
        
        t_a_plus = np.sqrt(a_plus**3 / mu) * (alpha_plus - beta_plus - np.sin(alpha_plus) + np.sin(beta_plus))
        
        # Slope of time vs. semi-major axis
        dt_da = (t_a_plus - t_a) / (a_plus - a)
        
        # Newton's method update
        a = a + (tof - t_a) / dt_da
        
        # Ensure a remains positive
        if a <= 0:
            a = a_min * 1.05
    
    # Calculate Lagrange coefficients
    f = 1 - r2_norm / a * (1 - np.cos(alpha))
    g = r1_norm * r2_norm * np.sin(alpha) / np.sqrt(mu * a)
    g_dot = 1 - r1_norm / a * (1 - np.cos(alpha))
    
    # Calculate velocity vectors
    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g
    
    return v1, v2

def multi_rev_lambert(r1, r2, tof, mu, n_revs, clockwise=False, max_iter=100, tol=1e-10):
    """
    Solve Lambert's problem with multiple revolutions.
    
    Args:
        r1: Initial position vector [x, y, z] (m)
        r2: Final position vector [x, y, z] (m)
        tof: Time of flight (s)
        mu: Gravitational parameter (m^3/s^2)
        n_revs: Number of complete revolutions
        clockwise: If True, choose clockwise motion (default: False)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
    
    Returns:
        Tuple of initial and final velocity vectors [v1, v2]
    """
    if n_revs <= 0:
        return solve_lambert(r1, r2, tof, mu, clockwise, max_iter, tol)
    
    r1 = np.array(r1)
    r2 = np.array(r2)
    
    # Magnitudes
    r1_norm = norm(r1)
    r2_norm = norm(r2)
    
    # Cross product of position vectors to determine orbital plane
    r1xr2 = np.cross(r1, r2)
    
    # If r1xr2 is zero, the points and the center are collinear
    if norm(r1xr2) < tol:
        # Handle the collinear case with a small perturbation
        r2 = r2 + np.array([1e-10, 1e-10, 1e-10])
        r1xr2 = np.cross(r1, r2)
    
    # Check if clockwise motion is requested
    if clockwise:
        r1xr2 = -r1xr2
    
    # Unit vector normal to the transfer plane
    h_hat = r1xr2 / norm(r1xr2)
    
    # Angle between position vectors (0 to pi)
    cos_dnu = np.dot(r1, r2) / (r1_norm * r2_norm)
    cos_dnu = np.clip(cos_dnu, -1.0, 1.0)  # Ensure domain -1 <= cos <= 1
    dnu = np.arccos(cos_dnu)
    
    # For multi-revolution, we always go the long way around
    cross_product = np.cross(r1, r2)
    if np.dot(r1xr2, cross_product) > 0:
        dnu = 2 * np.pi - dnu
    
    # Total angle traversed with multiple revolutions
    dnu_total = dnu + 2 * np.pi * n_revs
    
    # Semiperimeter and chord
    c = np.sqrt(r1_norm**2 + r2_norm**2 - 2 * r1_norm * r2_norm * np.cos(dnu))
    s = (r1_norm + r2_norm + c) / 2
    
    # Minimum energy ellipse semi-major axis
    a_min = s / 2
    
    # Initialize with a reasonable guess for multi-revolution case
    # For multiple revolutions, need a larger semi-major axis
    a = a_min * (1 + n_revs)
    
    # Solve for the semi-major axis using iterative approach
    for i in range(max_iter):
        # Calculate time of flight for current guess, including multiple revolutions
        alpha = 2 * np.arcsin(np.sqrt(s / (2 * a)))
        beta = 2 * np.arcsin(np.sqrt((s - c) / (2 * a)))
        
        if dnu > np.pi:
            beta = -beta
        
        t_a = np.sqrt(a**3 / mu) * (alpha - beta - np.sin(alpha) + np.sin(beta) + 2 * np.pi * n_revs)
        
        # Check convergence
        if abs(t_a - tof) < tol:
            break
        
        # Numerical differentiation for derivative
        a_plus = a * 1.01
        alpha_plus = 2 * np.arcsin(np.sqrt(s / (2 * a_plus)))
        beta_plus = 2 * np.arcsin(np.sqrt((s - c) / (2 * a_plus)))
        
        if dnu > np.pi:
            beta_plus = -beta_plus
        
        t_a_plus = np.sqrt(a_plus**3 / mu) * (alpha_plus - beta_plus - np.sin(alpha_plus) + np.sin(beta_plus) + 2 * np.pi * n_revs)
        
        # Slope of time vs. semi-major axis
        dt_da = (t_a_plus - t_a) / (a_plus - a)
        
        # Newton's method update
        a = a + (tof - t_a) / dt_da
        
        # Ensure a remains positive
        if a <= 0:
            a = a_min * (1 + n_revs) * 1.05
    
    # Calculate Lagrange coefficients
    f = 1 - r2_norm / a * (1 - np.cos(alpha))
    g = r1_norm * r2_norm * np.sin(alpha) / np.sqrt(mu * a)
    g_dot = 1 - r1_norm / a * (1 - np.cos(alpha))
    
    # Calculate velocity vectors
    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g
    
    return v1, v2 