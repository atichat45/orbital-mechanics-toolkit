#!/usr/bin/env python3
"""
Earth-to-Mars Transfer Trajectory Simulation

This script simulates a spacecraft transfer trajectory from Earth to Mars
using orbital elements and Kepler's equations, without relying on SPICE data for Mars.
"""

import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
import os
from datetime import datetime, timedelta
import math
from mpl_toolkits.mplot3d import Axes3D

# Standard gravitational parameters (GM) in km^3/s^2
SUN_MU = 1.32712440018e20
EARTH_MU = 3.986004418e14
MARS_MU = 4.282837e13

# Astronomical Unit in km
AU = 149597870.7

# Orbital elements of Earth and Mars (J2000)
# Semi-major axis (AU), eccentricity, inclination (deg), 
# longitude of ascending node (deg), argument of periapsis (deg),
# mean anomaly at epoch (deg), orbital period (days)
EARTH_ELEMENTS = {
    'a': 1.00000261,  # AU
    'e': 0.01671123,
    'i': 0.00005,     # almost zero
    'node': -11.26064,
    'peri': 102.94719,
    'M0': 100.46435,  # at J2000
    'period': 365.256363004  # days
}

MARS_ELEMENTS = {
    'a': 1.52366231,  # AU
    'e': 0.09341233,
    'i': 1.85061,
    'node': 49.57854,
    'peri': 336.04084,
    'M0': 355.45332,  # at J2000
    'period': 686.9800  # days
}

# Define epoch J2000
J2000 = "2000-01-01T12:00:00"

def load_spice_kernels():
    """Load necessary SPICE kernels."""
    print("Loading SPICE kernels...")
    
    kernels_dir = 'data/kernels'
    lsk_path = os.path.join(kernels_dir, 'naif0012.tls')
    spk_path = os.path.join(kernels_dir, 'de440.bsp')
    pck_path = os.path.join(kernels_dir, 'pck00010.tpc')
    
    # Load kernels
    spice.furnsh(lsk_path)
    spice.furnsh(spk_path)
    spice.furnsh(pck_path)
    
    return [lsk_path, spk_path, pck_path]

def unload_spice_kernels(kernel_paths):
    """Unload SPICE kernels."""
    print("Unloading SPICE kernels...")
    for kernel in kernel_paths:
        spice.unload(kernel)

def mean_anomaly_to_eccentric(M, e, tolerance=1e-10):
    """
    Convert mean anomaly to eccentric anomaly using Newton-Raphson method.
    
    Args:
        M: Mean anomaly in radians
        e: Eccentricity
        tolerance: Convergence tolerance
        
    Returns:
        Eccentric anomaly in radians
    """
    # Initial guess (improved for high eccentricity)
    if e < 0.8:
        E = M
    else:
        E = np.pi if M > np.pi else 0
        
    # Newton-Raphson iteration
    delta = 1.0
    iterations = 0
    max_iterations = 100
    
    while abs(delta) > tolerance and iterations < max_iterations:
        delta = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E = E - delta
        iterations += 1
        
    return E

def orbital_elements_to_state(elements, days_since_epoch):
    """
    Convert orbital elements to position and velocity vectors.
    
    Args:
        elements: Dictionary of orbital elements
        days_since_epoch: Days since J2000 epoch
        
    Returns:
        Tuple of position and velocity vectors in heliocentric frame
    """
    # Extract elements
    a = elements['a'] * AU  # Convert to km
    e = elements['e']
    i = np.radians(elements['i'])
    node = np.radians(elements['node'])
    peri = np.radians(elements['peri'])
    M0 = np.radians(elements['M0'])
    period = elements['period']
    
    # Calculate mean anomaly at the requested time
    n = 2 * np.pi / period  # Mean motion (rad/day)
    M = (M0 + n * days_since_epoch) % (2 * np.pi)
    
    # Calculate eccentric anomaly
    E = mean_anomaly_to_eccentric(M, e)
    
    # Calculate true anomaly
    cos_nu = (np.cos(E) - e) / (1 - e * np.cos(E))
    sin_nu = (np.sqrt(1 - e**2) * np.sin(E)) / (1 - e * np.cos(E))
    nu = np.arctan2(sin_nu, cos_nu)
    
    # Calculate distance from Sun
    r = a * (1 - e * np.cos(E))
    
    # Calculate position in orbital plane
    x_orbit = r * np.cos(nu)
    y_orbit = r * np.sin(nu)
    
    # Calculate velocity in orbital plane
    p = a * (1 - e**2)  # Semi-latus rectum
    vel_factor = np.sqrt(SUN_MU / p)
    vx_orbit = -vel_factor * np.sin(nu)
    vy_orbit = vel_factor * (e + np.cos(nu))
    
    # Rotation matrices
    # Rotation around z-axis by argument of periapsis
    R3_peri = np.array([
        [np.cos(peri), -np.sin(peri), 0],
        [np.sin(peri), np.cos(peri), 0],
        [0, 0, 1]
    ])
    
    # Rotation around x-axis by inclination
    R1_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]
    ])
    
    # Rotation around z-axis by longitude of ascending node
    R3_node = np.array([
        [np.cos(node), -np.sin(node), 0],
        [np.sin(node), np.cos(node), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix
    R = np.matmul(R3_node, np.matmul(R1_i, R3_peri))
    
    # Transform position and velocity to heliocentric ecliptic frame
    pos_orbital = np.array([x_orbit, y_orbit, 0])
    vel_orbital = np.array([vx_orbit, vy_orbit, 0])
    
    position = np.matmul(R, pos_orbital)
    velocity = np.matmul(R, vel_orbital)
    
    return position, velocity

def get_earth_state_at_time(date_str):
    """
    Get Earth's state vector (position and velocity) at a given time.
    Uses SPICE for Earth's position.
    
    Args:
        date_str: Date string in ISO format
        
    Returns:
        State vector [x, y, z, vx, vy, vz] in km and km/s
    """
    earth_id = 399  # SPICE ID for Earth
    sun_id = 10     # SPICE ID for Sun
    
    # Convert to ephemeris time
    et = spice.str2et(date_str)
    
    # Get the state vector from SPICE
    state, lt = spice.spkezr(
        str(earth_id), 
        et, 
        "ECLIPJ2000", 
        "NONE", 
        str(sun_id)
    )
    
    return np.array(state)

def get_mars_state_at_time(date_str):
    """
    Get Mars' state vector (position and velocity) at a given time.
    Uses orbital elements instead of SPICE.
    
    Args:
        date_str: Date string in ISO format
        
    Returns:
        State vector [x, y, z, vx, vy, vz] in km and km/s
    """
    # Convert to ephemeris time
    et = spice.str2et(date_str)
    
    # Calculate days since J2000
    j2000_et = spice.str2et(J2000)
    days_since_j2000 = (et - j2000_et) / 86400.0
    
    # Calculate state from orbital elements
    position, velocity = orbital_elements_to_state(MARS_ELEMENTS, days_since_j2000)
    
    return np.concatenate((position, velocity))

def lambert_solver(r1, r2, tof, mu=SUN_MU, clockwise=False):
    """
    Solves Lambert's problem for orbital determination.
    
    Args:
        r1: Initial position vector [x, y, z] in km
        r2: Final position vector [x, y, z] in km
        tof: Time of flight in seconds
        mu: Gravitational parameter in km^3/s^2
        clockwise: Whether the transfer is clockwise (True) or counter-clockwise (False)
        
    Returns:
        Tuple of initial and final velocity vectors [vx, vy, vz] in km/s
    """
    # Calculate the magnitudes of the position vectors
    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    
    # Calculate the cosine of the transfer angle
    r1_unit = r1 / r1_norm
    r2_unit = r2 / r2_norm
    cos_dnu = np.dot(r1_unit, r2_unit)
    
    # Ensure the angle is between 0 and π
    if cos_dnu > 1.0:
        cos_dnu = 1.0
    elif cos_dnu < -1.0:
        cos_dnu = -1.0
    
    # Calculate the transfer angle
    dnu = np.arccos(cos_dnu)
    
    # Adjust for clockwise or counter-clockwise transfer
    if np.cross(r1, r2)[2] < 0:
        clockwise_transfer = True
    else:
        clockwise_transfer = False
        
    if clockwise != clockwise_transfer:
        dnu = 2 * np.pi - dnu
    
    # Calculate the parameter of the transfer orbit
    A = np.sin(dnu) * np.sqrt(r1_norm * r2_norm / (1 - cos_dnu))
    
    # Use a simpler approach based on Battin's method and Gauss' solution
    # This is more reliable than the universal variable formulation
    
    # Check for 180-degree case (which can cause numerical issues)
    if abs(np.pi - dnu) < 1e-6:
        print("Warning: 180-degree transfer detected. Using specialized approach.")
        # For 180-degree transfer, we use a simple Hohmann transfer approximation
        a = (r1_norm + r2_norm) / 2
        
        # Calculate velocities at departure and arrival
        v1_mag = np.sqrt(mu * (2/r1_norm - 1/a))
        v2_mag = np.sqrt(mu * (2/r2_norm - 1/a))
        
        # Calculate velocity directions
        r1_cross_r2 = np.cross(r1, r2)
        h_unit = r1_cross_r2 / np.linalg.norm(r1_cross_r2)
        
        v1_dir = np.cross(h_unit, r1_unit)
        v2_dir = np.cross(h_unit, r2_unit)
        
        if clockwise:
            v1_dir = -v1_dir
            v2_dir = -v2_dir
            
        v1 = v1_mag * v1_dir
        v2 = v2_mag * v2_dir
        
        return v1, v2
    
    # For typical transfers, use Battin's method
    # Direct implementation inspired by Vallado's "Fundamentals of Astrodynamics and Applications"
    
    # Calculate chord and semi-perimeter
    c = np.sqrt(r1_norm**2 + r2_norm**2 - 2 * r1_norm * r2_norm * cos_dnu)
    s = (r1_norm + r2_norm + c) / 2
    
    # Determine the minimum energy orbit's semi-major axis
    a_min = s / 2
    
    # Calculate orbital parameters for the transfer
    # First guess: slightly higher energy than minimum (typical for realistic transfers)
    a = a_min * 1.1
    
    # Calculate eccentricity and semi-latus rectum
    e = np.sqrt(1 - (2 * r1_norm * r2_norm * np.sin(dnu/2)**2) / (a * s))
    p = a * (1 - e**2)
    
    # Iteratively refine to match time of flight
    tol = 1e-8
    max_iter = 100
    iter_count = 0
    converged = False
    
    while not converged and iter_count < max_iter:
        iter_count += 1
        
        # Calculate period for current semi-major axis
        if a > 0:  # Elliptical orbit
            period = 2 * np.pi * np.sqrt(a**3 / mu)
            
            # Calculate eccentric anomaly change
            sin_E_half = np.sqrt((s - c) / (2 * a))
            cos_E_half = np.sqrt((s) / (2 * a))
            E_change = 2 * np.arctan2(sin_E_half, cos_E_half)
            
            # Calculate time of flight
            dt = period * (E_change - e * np.sin(E_change)) / (2 * np.pi)
            
        else:  # Hyperbolic orbit
            # Calculate hyperbolic anomaly change
            sin_H_half = np.sqrt((c - s) / (-2 * a))
            cos_H_half = np.sqrt((-s) / (-2 * a))
            H_change = 2 * np.arcsinh(sin_H_half / cos_H_half)
            
            # Calculate time of flight
            dt = np.sqrt(-a**3 / mu) * (e * np.sinh(H_change) - H_change)
        
        # Check convergence
        if abs(dt - tof) < tol:
            converged = True
            break
        
        # Update semi-major axis using secant method
        if iter_count == 1:
            a_prev = a
            dt_prev = dt
            if dt < tof:
                a = a * 1.1  # Increase energy if flight time too short
            else:
                a = a * 0.9  # Decrease energy if flight time too long
        else:
            # Secant method for faster convergence
            a_new = a - (dt - tof) * (a - a_prev) / (dt - dt_prev)
            a_prev = a
            dt_prev = dt
            a = a_new
        
        # Update eccentricity and semi-latus rectum
        e = np.sqrt(1 - (2 * r1_norm * r2_norm * np.sin(dnu/2)**2) / (a * s))
        p = a * (1 - e**2)
    
    if not converged and iter_count >= max_iter:
        print(f"Warning: Lambert solver did not converge after {max_iter} iterations. Using last estimate.")
    
    # Calculate Lagrange coefficients
    if a > 0:  # Elliptical orbit
        # Calculate specific angular momentum
        h = np.sqrt(mu * p)
        
        # Calculate Lagrange coefficients
        f = 1.0 - (r2_norm / p) * (1.0 - np.cos(dnu))
        g = (r1_norm * r2_norm * np.sin(dnu)) / np.sqrt(mu * p)
        gdot = 1.0 - (r1_norm / p) * (1.0 - np.cos(dnu))
    else:  # Hyperbolic orbit
        # Similar calculations but with appropriate adjustments
        h = np.sqrt(mu * p)
        f = 1.0 - (r2_norm / p) * (1.0 - np.cos(dnu))
        g = (r1_norm * r2_norm * np.sin(dnu)) / np.sqrt(mu * p)
        gdot = 1.0 - (r1_norm / p) * (1.0 - np.cos(dnu))
    
    # Calculate velocities using Lagrange coefficients
    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    
    return v1, v2

def simulate_transfer_trajectory(departure_date, tof_days):
    """
    Simulate an Earth-to-Mars transfer trajectory.
    
    Args:
        departure_date: Departure date (ISO format)
        tof_days: Time of flight in days
        
    Returns:
        Dictionary with trajectory information
    """
    # Convert time of flight to seconds
    seconds_per_day = 86400
    tof_seconds = tof_days * seconds_per_day
    
    # Calculate arrival date
    et_departure = spice.str2et(departure_date)
    et_arrival = et_departure + tof_seconds
    arrival_date = spice.et2utc(et_arrival, "ISOC", 0)
    
    print(f"Simulating transfer trajectory from Earth on {departure_date} to Mars on {arrival_date}...")
    print(f"Time of flight: {tof_days} days")
    
    # Get Earth state at departure (using SPICE)
    earth_state_dep = get_earth_state_at_time(departure_date)
    
    # Get Mars state at arrival (using orbital elements)
    mars_state_arr = get_mars_state_at_time(arrival_date)
    
    # Extract positions and velocities
    r1 = earth_state_dep[:3]  # Earth position at departure
    v1_earth = earth_state_dep[3:]  # Earth velocity at departure
    r2 = mars_state_arr[:3]  # Mars position at arrival
    v2_mars = mars_state_arr[3:]  # Mars velocity at arrival
    
    # Solve Lambert's problem for transfer orbit
    v1_trans, v2_trans = lambert_solver(r1, r2, tof_seconds, mu=SUN_MU)
    
    # Calculate delta-Vs
    delta_v1 = np.linalg.norm(v1_trans - v1_earth)
    delta_v2 = np.linalg.norm(v2_trans - v2_mars)
    total_delta_v = delta_v1 + delta_v2
    
    # Calculate C3 (characteristic energy)
    c3 = np.linalg.norm(v1_trans)**2 - 2*EARTH_MU/np.linalg.norm(r1)
    
    print(f"Transfer Orbit Parameters:")
    print(f"  Departure: {departure_date}")
    print(f"  Arrival: {arrival_date}")
    print(f"  Time of Flight: {tof_days} days")
    print(f"  Earth Departure ΔV: {delta_v1:.2f} km/s")
    print(f"  Mars Arrival ΔV: {delta_v2:.2f} km/s")
    print(f"  Total ΔV: {total_delta_v:.2f} km/s")
    print(f"  C3: {c3:.2f} km²/s²")
    
    # Propagate spacecraft's trajectory
    n_points = 500
    traj_times = np.linspace(0, tof_seconds, n_points)
    
    # Propagate using two-body equation
    sc_trajectory = propagate_two_body(r1, v1_trans, traj_times, SUN_MU)
    
    # Get Earth and Mars trajectories over the transfer period
    earth_trajectory = np.zeros((n_points, 3))
    mars_trajectory = np.zeros((n_points, 3))
    
    for i, dt in enumerate(traj_times):
        time_et = et_departure + dt
        date_str = spice.et2utc(time_et, "ISOC", 0)
        
        # Get Earth state at each point (using SPICE)
        earth_state = get_earth_state_at_time(date_str)
        earth_trajectory[i] = earth_state[:3]
        
        # Get Mars state at each point (using orbital elements)
        mars_state = get_mars_state_at_time(date_str)
        mars_trajectory[i] = mars_state[:3]
    
    # Compile trajectory information
    trajectory_info = {
        'departure_date': departure_date,
        'arrival_date': arrival_date,
        'tof_seconds': tof_seconds,
        'tof_days': tof_days,
        'delta_v_departure': delta_v1,
        'delta_v_arrival': delta_v2,
        'total_delta_v': total_delta_v,
        'c3': c3,
        'r1': r1,
        'v1_earth': v1_earth,
        'v1_trans': v1_trans,
        'r2': r2,
        'v2_mars': v2_mars,
        'v2_trans': v2_trans,
        'sc_trajectory': sc_trajectory,
        'earth_trajectory': earth_trajectory,
        'mars_trajectory': mars_trajectory,
        'times': np.linspace(et_departure, et_arrival, n_points)
    }
    
    return trajectory_info

def propagate_two_body(r0, v0, dt_array, mu):
    """
    Propagate a trajectory using the two-body equation.
    
    Args:
        r0: Initial position vector [x, y, z] in km
        v0: Initial velocity vector [vx, vy, vz] in km/s
        dt_array: Array of time steps in seconds
        mu: Gravitational parameter in km^3/s^2
        
    Returns:
        Array of position vectors at each time step
    """
    # Initialize result array
    n_points = len(dt_array)
    trajectory = np.zeros((n_points, 3))
    
    # Calculate initial orbital elements
    h = np.cross(r0, v0)  # Angular momentum vector
    r_norm = np.linalg.norm(r0)
    v_norm = np.linalg.norm(v0)
    
    # Semi-major axis
    a = 1.0 / (2.0 / r_norm - v_norm**2 / mu)
    
    # Eccentricity vector
    e_vec = np.cross(v0, h) / mu - r0 / r_norm
    e = np.linalg.norm(e_vec)
    
    # Check for special cases
    if abs(e - 1.0) < 1e-10:  # Parabolic orbit
        print("Warning: Parabolic orbit detected (e=1). Using approximation.")
        e = 0.999  # Use near-parabolic approximation
    
    # Semi-latus rectum
    p = a * (1 - e**2)
    
    # Mean motion
    n = np.sqrt(mu / abs(a)**3)
    
    # Calculate initial true anomaly
    r_dot_e = np.dot(r0, e_vec)
    cos_nu0 = r_dot_e / (r_norm * e)
    if cos_nu0 > 1.0:
        cos_nu0 = 1.0
    elif cos_nu0 < -1.0:
        cos_nu0 = -1.0
    
    nu0 = np.arccos(cos_nu0)
    if np.dot(r0, v0) < 0:
        nu0 = 2 * np.pi - nu0
    
    # Calculate initial eccentric anomaly
    if e < 1.0:  # Elliptic orbit
        cos_E0 = (e + np.cos(nu0)) / (1 + e * np.cos(nu0))
        sin_E0 = np.sqrt(1 - e**2) * np.sin(nu0) / (1 + e * np.cos(nu0))
        E0 = np.arctan2(sin_E0, cos_E0)
        
        # Initial mean anomaly
        M0 = E0 - e * np.sin(E0)
    else:  # Hyperbolic orbit
        cos_F0 = (e + np.cos(nu0)) / (1 + e * np.cos(nu0))
        sin_F0 = np.sqrt(e**2 - 1) * np.sin(nu0) / (1 + e * np.cos(nu0))
        F0 = np.arctan2(sin_F0, cos_F0)
        
        # Initial mean anomaly
        M0 = e * np.sinh(F0) - F0
    
    # Propagate trajectory for each time step
    for i, dt in enumerate(dt_array):
        # Calculate mean anomaly at time t
        M = M0 + n * dt
        
        # Calculate eccentric anomaly at time t
        if e < 1.0:  # Elliptic orbit
            E = M
            # Newton-Raphson iteration to solve Kepler's equation
            for j in range(20):
                E_next = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
                if abs(E_next - E) < 1e-10:
                    break
                E = E_next
            
            # Calculate true anomaly
            cos_nu = (np.cos(E) - e) / (1 - e * np.cos(E))
            sin_nu = np.sqrt(1 - e**2) * np.sin(E) / (1 - e * np.cos(E))
            nu = np.arctan2(sin_nu, cos_nu)
        else:  # Hyperbolic orbit
            # Initial guess for hyperbolic anomaly
            if M > 0:
                F = np.log(2 * M / e)
            else:
                F = -np.log(-2 * M / e)
            
            # Newton-Raphson iteration to solve Kepler's equation
            for j in range(20):
                F_next = F - (e * np.sinh(F) - F - M) / (e * np.cosh(F) - 1)
                if abs(F_next - F) < 1e-10:
                    break
                F = F_next
            
            # Calculate true anomaly
            cos_nu = (np.cosh(F) - e) / (1 - e * np.cosh(F))
            sin_nu = -np.sqrt(e**2 - 1) * np.sinh(F) / (1 - e * np.cosh(F))
            nu = np.arctan2(sin_nu, cos_nu)
        
        # Calculate distance from central body
        r = p / (1 + e * np.cos(nu))
        
        # Rotate position vector from perifocal to inertial frame
        # First, define perifocal frame vectors
        h_unit = h / np.linalg.norm(h)
        e_unit = e_vec / e
        n_unit = np.cross(e_unit, h_unit)
        
        # Position in perifocal frame
        x_peri = r * np.cos(nu)
        y_peri = r * np.sin(nu)
        
        # Transform to inertial frame
        trajectory[i] = x_peri * e_unit + y_peri * n_unit
    
    return trajectory

def plot_transfer_trajectory(trajectory_info, output_file="earth_mars_transfer.png"):
    """
    Plot the Earth-to-Mars transfer trajectory.
    
    Args:
        trajectory_info: Dictionary with trajectory information
        output_file: File to save the plot
    """
    print("Generating transfer trajectory plot...")
    
    # Extract data
    sc_trajectory = trajectory_info['sc_trajectory']
    earth_trajectory = trajectory_info['earth_trajectory']
    mars_trajectory = trajectory_info['mars_trajectory']
    r1 = trajectory_info['r1']
    r2 = trajectory_info['r2']
    
    # Create figure
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    ax.plot(earth_trajectory[:, 0], earth_trajectory[:, 1], earth_trajectory[:, 2], 
            'b-', label='Earth Orbit', linewidth=2, alpha=0.7)
    ax.plot(mars_trajectory[:, 0], mars_trajectory[:, 1], mars_trajectory[:, 2], 
            'r-', label='Mars Orbit', linewidth=2, alpha=0.7)
    ax.plot(sc_trajectory[:, 0], sc_trajectory[:, 1], sc_trajectory[:, 2], 
            'g--', label='Spacecraft Trajectory', linewidth=2)
    
    # Plot planets at departure and arrival
    ax.scatter(0, 0, 0, color='gold', s=300, edgecolor='black', label='Sun')
    ax.scatter(r1[0], r1[1], r1[2], color='royalblue', s=150, edgecolor='black', label='Earth at Departure')
    ax.scatter(r2[0], r2[1], r2[2], color='firebrick', s=100, edgecolor='black', label='Mars at Arrival')
    
    # Mark transfer trajectory
    ax.scatter(sc_trajectory[0, 0], sc_trajectory[0, 1], sc_trajectory[0, 2], 
               color='green', s=80, marker='^', edgecolor='black', label='Departure')
    ax.scatter(sc_trajectory[-1, 0], sc_trajectory[-1, 1], sc_trajectory[-1, 2], 
               color='green', s=80, marker='v', edgecolor='black', label='Arrival')
    
    # Draw radial lines to planets
    ax.plot([0, r1[0]], [0, r1[1]], [0, r1[2]], 'k:', alpha=0.3)
    ax.plot([0, r2[0]], [0, r2[1]], [0, r2[2]], 'k:', alpha=0.3)
    
    # Add orbital plane visualization
    orbital_plane_points = np.vstack([np.zeros(3), r1, r2])
    hull = ax.plot_trisurf(orbital_plane_points[:, 0], 
                          orbital_plane_points[:, 1], 
                          orbital_plane_points[:, 2],
                          alpha=0.1, color='gray')
    
    # Set equal aspect ratio
    max_val = max(
        np.max(np.abs(earth_trajectory)),
        np.max(np.abs(mars_trajectory)),
        np.max(np.abs(sc_trajectory))
    )
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_zlim(-max_val, max_val)
    
    # Labels and title
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Earth-to-Mars Interplanetary Transfer Trajectory')
    
    # Add mission info
    mission_info = (
        f"Mission Parameters:\n"
        f"Departure: {trajectory_info['departure_date']}\n"
        f"Arrival: {trajectory_info['arrival_date']}\n"
        f"Time of Flight: {trajectory_info['tof_days']} days\n"
        f"Earth Departure ΔV: {trajectory_info['delta_v_departure']:.2f} km/s\n"
        f"Mars Arrival ΔV: {trajectory_info['delta_v_arrival']:.2f} km/s\n"
        f"Total ΔV: {trajectory_info['total_delta_v']:.2f} km/s\n"
        f"C3: {trajectory_info['c3']:.2f} km²/s²"
    )
    plt.figtext(0.15, 0.05, mission_info, fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.7))
    
    # Add legend and adjust view
    ax.legend(loc='upper right')
    ax.view_init(elev=30, azim=45)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Transfer trajectory plot saved to {output_file}")

def analyze_mission_parameters(trajectory_info):
    """
    Analyze mission parameters and print detailed information.
    
    Args:
        trajectory_info: Dictionary with trajectory information
    """
    print("\n===== MISSION PARAMETER ANALYSIS =====")
    
    # Extract basic parameters
    departure_date = trajectory_info['departure_date']
    arrival_date = trajectory_info['arrival_date']
    tof_days = trajectory_info['tof_days']
    delta_v_dep = trajectory_info['delta_v_departure']
    delta_v_arr = trajectory_info['delta_v_arrival']
    total_delta_v = trajectory_info['total_delta_v']
    c3 = trajectory_info['c3']
    
    # Earth departure parameters
    r1 = trajectory_info['r1']
    v1_earth = trajectory_info['v1_earth']
    v1_trans = trajectory_info['v1_trans']
    
    # Mars arrival parameters
    r2 = trajectory_info['r2']
    v2_mars = trajectory_info['v2_mars']
    v2_trans = trajectory_info['v2_trans']
    
    # Calculate mission phases
    
    # 1. Earth parking orbit parameters (assume circular 300 km altitude)
    earth_radius = 6378.0  # km
    parking_alt = 300.0  # km
    parking_radius = earth_radius + parking_alt
    parking_velocity = math.sqrt(EARTH_MU / parking_radius)  # km/s
    
    # 2. Trans-Mars injection (TMI) burn
    # Conversion from Earth-centered velocity to heliocentric
    hyperbolic_excess_velocity = np.linalg.norm(v1_trans - v1_earth)  # km/s
    tmi_delta_v = math.sqrt(hyperbolic_excess_velocity**2 + 2*EARTH_MU/parking_radius) - parking_velocity
    
    # 3. Mars orbit insertion (MOI) burn (assume circular 400 km altitude)
    mars_radius = 3396.0  # km
    mars_orbit_alt = 400.0  # km
    mars_orbit_radius = mars_radius + mars_orbit_alt
    mars_orbit_velocity = math.sqrt(MARS_MU / mars_orbit_radius)  # km/s
    
    # Calculate hyperbolic approach velocity at Mars
    hyperbolic_approach_velocity = np.linalg.norm(v2_trans - v2_mars)  # km/s
    moi_delta_v = math.sqrt(hyperbolic_approach_velocity**2 + 2*MARS_MU/mars_orbit_radius) - mars_orbit_velocity
    
    # 4. Transfer orbit characteristics
    sc_trajectory = trajectory_info['sc_trajectory']
    min_sun_dist = min(np.linalg.norm(pos) for pos in sc_trajectory)
    max_sun_dist = max(np.linalg.norm(pos) for pos in sc_trajectory)
    transfer_a = (min_sun_dist + max_sun_dist) / 2
    transfer_e = (max_sun_dist - min_sun_dist) / (max_sun_dist + min_sun_dist)
    
    # Print detailed mission analysis
    print("\nMission Timeline:")
    print(f"  Earth Departure Date: {departure_date}")
    print(f"  Mars Arrival Date: {arrival_date}")
    print(f"  Time of Flight: {tof_days} days")
    
    print("\nTransfer Orbit Characteristics:")
    print(f"  Semi-major Axis: {transfer_a:.2f} km")
    print(f"  Eccentricity: {transfer_e:.4f}")
    print(f"  Perihelion Distance: {min_sun_dist:.2f} km")
    print(f"  Aphelion Distance: {max_sun_dist:.2f} km")
    
    print("\nEarth Departure:")
    print(f"  Earth Parking Orbit Altitude: {parking_alt:.1f} km")
    print(f"  Parking Orbit Velocity: {parking_velocity:.2f} km/s")
    print(f"  Hyperbolic Excess Velocity: {hyperbolic_excess_velocity:.2f} km/s")
    print(f"  C3 (Characteristic Energy): {c3:.2f} km²/s²")
    print(f"  TMI Delta-V (from parking orbit): {tmi_delta_v:.2f} km/s")
    
    print("\nMars Arrival:")
    print(f"  Mars Orbit Altitude: {mars_orbit_alt:.1f} km")
    print(f"  Mars Orbit Velocity: {mars_orbit_velocity:.2f} km/s")
    print(f"  Hyperbolic Approach Velocity: {hyperbolic_approach_velocity:.2f} km/s")
    print(f"  MOI Delta-V (for capture): {moi_delta_v:.2f} km/s")
    
    print("\nTotal Mission Delta-V:")
    print(f"  TMI (Earth Departure): {tmi_delta_v:.2f} km/s")
    print(f"  MOI (Mars Capture): {moi_delta_v:.2f} km/s")
    print(f"  Total Required: {tmi_delta_v + moi_delta_v:.2f} km/s")
    
    # Return mission analysis data
    analysis = {
        'tmi_delta_v': tmi_delta_v,
        'moi_delta_v': moi_delta_v,
        'total_mission_delta_v': tmi_delta_v + moi_delta_v,
        'transfer_a': transfer_a,
        'transfer_e': transfer_e,
        'min_sun_dist': min_sun_dist,
        'max_sun_dist': max_sun_dist,
        'parking_orbit_velocity': parking_velocity,
        'hyperbolic_excess_velocity': hyperbolic_excess_velocity,
        'mars_orbit_velocity': mars_orbit_velocity,
        'hyperbolic_approach_velocity': hyperbolic_approach_velocity
    }
    
    return analysis

def main():
    """Main function to run the Earth-to-Mars mission simulation."""
    try:
        # Load SPICE kernels
        kernel_paths = load_spice_kernels()
        
        # Set fixed mission parameters for a typical Hohmann-like transfer
        departure_date = "2022-01-20T00:00:00"  # Earth departure date
        tof_days = 259  # Typical Earth-to-Mars transfer time (~8.5 months)
        
        # Simulate transfer trajectory with fixed parameters
        trajectory = simulate_transfer_trajectory(departure_date, tof_days)
        
        # Analyze mission parameters
        mission_analysis = analyze_mission_parameters(trajectory)
        
        # Create visualization
        plot_transfer_trajectory(trajectory, "earth_mars_transfer.png")
        
        print("\nEarth-to-Mars mission simulation completed successfully!")
        
    except Exception as e:
        print(f"Error in mission simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Unload SPICE kernels
        unload_spice_kernels(kernel_paths)

if __name__ == "__main__":
    main() 