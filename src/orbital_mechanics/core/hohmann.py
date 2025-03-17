"""
Hohmann Transfer Orbit Calculations

This module provides functions for calculating Hohmann transfer orbits between two circular orbits.
It includes analytical and numerical approaches for computing transfer parameters.
"""

import numpy as np
from scipy.integrate import solve_ivp

# Constants
G = 6.67430e-11  # Gravitational constant, m^3/kg/s^2
M_SUN = 1.989e30  # Solar mass, kg
AU = 149.6e9  # Astronomical unit, m
DAY = 86400  # Seconds in a day
YEAR = 365.25 * DAY  # Seconds in a year

# Convert gravitational parameter to m^3/s^2
MU_SUN = G * M_SUN

def hohmann_transfer_analytical(r1, r2, mu=MU_SUN):
    """
    Calculate Hohmann transfer orbit parameters analytically.
    
    Args:
        r1: Radius of departure circular orbit (m)
        r2: Radius of arrival circular orbit (m)
        mu: Gravitational parameter (m^3/s^2)
    
    Returns:
        Dictionary with transfer parameters
    """
    # Validate inputs
    if r1 <= 0 or r2 <= 0:
        raise ValueError("Orbit radii must be positive")
    
    # Semi-major axis of the transfer orbit
    a_transfer = (r1 + r2) / 2
    
    # Velocity in circular orbit at r1
    v1_circular = np.sqrt(mu / r1)
    
    # Velocity in circular orbit at r2
    v2_circular = np.sqrt(mu / r2)
    
    # Velocity at periapsis of transfer orbit
    v1_transfer = np.sqrt(mu * (2/r1 - 1/a_transfer))
    
    # Velocity at apoapsis of transfer orbit
    v2_transfer = np.sqrt(mu * (2/r2 - 1/a_transfer))
    
    # Delta-V at departure
    delta_v_departure = abs(v1_transfer - v1_circular)
    
    # Delta-V at arrival
    delta_v_arrival = abs(v2_circular - v2_transfer)
    
    # Total Delta-V
    delta_v_total = delta_v_departure + delta_v_arrival
    
    # Time of flight (half-orbit)
    tof = np.pi * np.sqrt(a_transfer**3 / mu)
    
    # Path length (semi-ellipse)
    e = abs(r2 - r1) / (r2 + r1)  # eccentricity
    a = a_transfer
    b = a * np.sqrt(1 - e**2)  # semi-minor axis
    
    # More accurate approximation of ellipse perimeter using Ramanujan's formula
    h = ((a - b) / (a + b))**2
    path_length = np.pi * (a + b) * (1 + 3*h/(10 + np.sqrt(4 - 3*h)))
    
    # Calculate characteristic energy (C3)
    c3 = v1_transfer**2 - 2*mu/r1
    
    return {
        'a_transfer': a_transfer,
        'e_transfer': e,
        'delta_v_departure': delta_v_departure,
        'delta_v_arrival': delta_v_arrival,
        'delta_v_total': delta_v_total,
        'tof': tof,
        'path_length': path_length,
        'c3': c3
    }

def two_body_equation(t, y, mu):
    """
    Two-body equation of motion for numerical integration.
    
    Args:
        t: Time (not used, required by solve_ivp)
        y: State vector [x, y, z, vx, vy, vz]
        mu: Gravitational parameter
    
    Returns:
        Derivatives [vx, vy, vz, ax, ay, az]
    """
    x, y, z, vx, vy, vz = y
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Check for division by zero
    if r < 1e-10:
        r = 1e-10
    
    # Acceleration components
    ax = -mu * x / r**3
    ay = -mu * y / r**3
    az = -mu * z / r**3
    
    return [vx, vy, vz, ax, ay, az]

def propagate_orbit_numerical(r0, v0, tof, mu=MU_SUN, steps=1000):
    """
    Numerically propagate an orbit from initial state for specified time.
    
    Args:
        r0: Initial position vector [x, y, z] (m)
        v0: Initial velocity vector [vx, vy, vz] (m/s)
        tof: Time of flight (s)
        mu: Gravitational parameter (m^3/s^2)
        steps: Number of steps for solution
    
    Returns:
        Dictionary with trajectory data or None if propagation fails
    """
    # Initial state vector
    y0 = np.concatenate((r0, v0))
    
    # Time span
    t_span = (0, tof)
    t_eval = np.linspace(0, tof, steps)
    
    try:
        # Numerical integration
        solution = solve_ivp(
            two_body_equation, 
            t_span, 
            y0, 
            args=(mu,), 
            method='RK45', 
            t_eval=t_eval,
            rtol=1e-10, 
            atol=1e-10
        )
        
        # Extract trajectory data
        trajectory = solution.y.T
        times = solution.t
        
        # Check if the solution was successful
        if not solution.success:
            print(f"Warning: Integration was not successful: {solution.message}")
            
        # Calculate path length (sum of segment lengths)
        positions = trajectory[:, 0:3]
        segments = np.diff(positions, axis=0)
        segment_lengths = np.sqrt(np.sum(segments**2, axis=1))
        path_length = np.sum(segment_lengths)
        
        # Calculate final state
        r_final = trajectory[-1, 0:3]
        v_final = trajectory[-1, 3:6]
        
        return {
            'trajectory': trajectory,
            'times': times,
            'path_length': path_length,
            'r_final': r_final,
            'v_final': v_final,
            'success': solution.success,
            'message': solution.message
        }
    except Exception as e:
        print(f"Error in orbit propagation: {str(e)}")
        return None 