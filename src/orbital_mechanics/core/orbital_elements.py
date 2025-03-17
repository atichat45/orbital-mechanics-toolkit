"""
Orbital Elements and Transformations

This module provides functions for converting between different representations
of orbits, including Cartesian state vectors and Keplerian orbital elements.
"""

import numpy as np

def cartesian_to_keplerian(r, v, mu):
    """
    Convert Cartesian state vectors to Keplerian orbital elements.
    
    Args:
        r: Position vector [x, y, z] (m)
        v: Velocity vector [vx, vy, vz] (m/s)
        mu: Gravitational parameter (m^3/s^2)
    
    Returns:
        Dictionary containing Keplerian elements:
            a: Semi-major axis (m)
            e: Eccentricity (unitless)
            i: Inclination (rad)
            Omega: Right ascension of ascending node (rad)
            omega: Argument of periapsis (rad)
            nu: True anomaly (rad)
    """
    r = np.array(r)
    v = np.array(v)
    
    # Magnitude of position and velocity
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    
    # Specific angular momentum
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    
    # Node line
    k = np.array([0, 0, 1])
    n = np.cross(k, h)
    n_norm = np.linalg.norm(n)
    
    # Eccentricity vector
    e_vec = ((v_norm**2 - mu/r_norm) * r - np.dot(r, v) * v) / mu
    e = np.linalg.norm(e_vec)
    
    # Semi-major axis
    if abs(e - 1.0) < 1e-10:  # Parabolic orbit
        a = float('inf')
    else:
        a = h_norm**2 / (mu * (1 - e**2))
    
    # Inclination
    i = np.arccos(h[2] / h_norm)
    
    # Right ascension of ascending node
    if n_norm < 1e-10:  # Equatorial orbit
        Omega = 0.0
    else:
        Omega = np.arccos(n[0] / n_norm)
        if n[1] < 0:
            Omega = 2 * np.pi - Omega
    
    # Argument of periapsis
    if n_norm < 1e-10:  # Equatorial orbit
        omega = np.arctan2(e_vec[1], e_vec[0])
    else:
        omega = np.arccos(np.dot(n, e_vec) / (n_norm * e))
        if e_vec[2] < 0:
            omega = 2 * np.pi - omega
    
    # True anomaly
    if e < 1e-10:  # Circular orbit
        # Fix for circular orbits - use the position and velocity vectors directly
        if n_norm < 1e-10:  # Circular equatorial orbit
            # For circular equatorial orbits, define the true anomaly from x-axis
            nu = np.arctan2(r[1], r[0])
        else:
            # For circular inclined orbits, true anomaly is angle from ascending node
            nu = np.arccos(np.dot(n, r) / (n_norm * r_norm))
            if np.dot(n, r) < 0 or np.dot(np.cross(n, r), h) < 0:
                nu = 2 * np.pi - nu
    else:
        nu = np.arccos(np.dot(e_vec, r) / (e * r_norm))
        if np.dot(r, v) < 0:
            nu = 2 * np.pi - nu
    
    return {
        'a': a,
        'e': e,
        'i': i,
        'Omega': Omega,
        'omega': omega,
        'nu': nu
    }

def keplerian_to_cartesian(elements, mu):
    """
    Convert Keplerian orbital elements to Cartesian state vectors.
    
    Args:
        elements: Dictionary containing Keplerian elements:
            a: Semi-major axis (m)
            e: Eccentricity (unitless)
            i: Inclination (rad)
            Omega: Right ascension of ascending node (rad)
            omega: Argument of periapsis (rad)
            nu: True anomaly (rad)
        mu: Gravitational parameter (m^3/s^2)
    
    Returns:
        Tuple of position and velocity vectors [r, v]
    """
    a = elements['a']
    e = elements['e']
    i = elements['i']
    Omega = elements['Omega']
    omega = elements['omega']
    nu = elements['nu']
    
    # Semi-latus rectum
    p = a * (1 - e**2)
    
    # Position in orbital plane
    r_orb = np.array([
        p * np.cos(nu) / (1 + e * np.cos(nu)),
        p * np.sin(nu) / (1 + e * np.cos(nu)),
        0
    ])
    
    # Velocity in orbital plane
    v_orb = np.array([
        -np.sqrt(mu / p) * np.sin(nu),
        np.sqrt(mu / p) * (e + np.cos(nu)),
        0
    ])
    
    # Rotation matrices
    R3_Omega = np.array([
        [np.cos(Omega), -np.sin(Omega), 0],
        [np.sin(Omega), np.cos(Omega), 0],
        [0, 0, 1]
    ])
    
    R1_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]
    ])
    
    R3_omega = np.array([
        [np.cos(omega), -np.sin(omega), 0],
        [np.sin(omega), np.cos(omega), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = np.dot(R3_Omega, np.dot(R1_i, R3_omega))
    
    # Transform to inertial frame
    r = np.dot(R, r_orb)
    v = np.dot(R, v_orb)
    
    return r, v

def mean_to_eccentric_anomaly(M, e, tolerance=1e-10, max_iterations=100):
    """
    Convert mean anomaly to eccentric anomaly using Newton-Raphson method.
    
    Args:
        M: Mean anomaly (rad)
        e: Eccentricity (unitless)
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations
    
    Returns:
        Eccentric anomaly (rad)
    """
    # Initial guess
    if e < 0.8:
        E = M
    else:
        E = np.pi
    
    # Newton-Raphson iteration
    for i in range(max_iterations):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        
        delta_E = f / f_prime
        E = E - delta_E
        
        if abs(delta_E) < tolerance:
            break
    
    return E

def eccentric_to_true_anomaly(E, e):
    """
    Convert eccentric anomaly to true anomaly.
    
    Args:
        E: Eccentric anomaly (rad)
        e: Eccentricity (unitless)
    
    Returns:
        True anomaly (rad)
    """
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    return nu % (2 * np.pi)

def true_to_eccentric_anomaly(nu, e):
    """
    Convert true anomaly to eccentric anomaly.
    
    Args:
        nu: True anomaly (rad)
        e: Eccentricity (unitless)
    
    Returns:
        Eccentric anomaly (rad)
    """
    E = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2), np.sqrt(1 + e) * np.cos(nu / 2))
    return E % (2 * np.pi)

def eccentric_to_mean_anomaly(E, e):
    """
    Convert eccentric anomaly to mean anomaly.
    
    Args:
        E: Eccentric anomaly (rad)
        e: Eccentricity (unitless)
    
    Returns:
        Mean anomaly (rad)
    """
    M = E - e * np.sin(E)
    return M % (2 * np.pi) 