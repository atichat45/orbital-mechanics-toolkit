#!/usr/bin/env python3
"""
Simple Earth-to-Mars Transfer Trajectory Simulation

This script simulates a spacecraft transfer trajectory from Earth to Mars
using basic orbital mechanics and simplified models.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import math
import os
import datetime

# Constants (in standard units)
AU = 149.6e6  # Astronomical Unit in km
SUN_MU = 1.32712440018e20  # Sun's gravitational parameter in km^3/s^2
EARTH_MU = 3.986004418e14  # Earth's gravitational parameter in km^3/s^2
MARS_MU = 4.282837e13  # Mars's gravitational parameter in km^3/s^2
DAY = 86400  # Seconds in a day

# Orbital elements (simplified, mean elements)
EARTH_ORBIT = {
    'a': 1.00000261 * AU,  # Semi-major axis (km)
    'e': 0.01671123,       # Eccentricity
    'i': 0.00005,          # Inclination (rad)
    'P': 365.256363004 * DAY  # Orbital period (seconds)
}

MARS_ORBIT = {
    'a': 1.52366231 * AU,  # Semi-major axis (km)
    'e': 0.09341233,       # Eccentricity
    'i': 0.03232 ,         # Inclination (rad)
    'P': 686.9800 * DAY    # Orbital period (seconds)
}

def get_position_at_time(orbit, t, mu=SUN_MU):
    """
    Calculate position of a body in its orbit at time t.
    
    Args:
        orbit: Dictionary with orbital elements
        t: Time in seconds
        mu: Gravitational parameter of central body
    
    Returns:
        Position vector [x, y, z] in km
    """
    a = orbit['a']  # Semi-major axis
    e = orbit['e']  # Eccentricity
    i = orbit['i']  # Inclination
    P = orbit['P']  # Period
    
    # Calculate mean anomaly at time t
    n = 2 * np.pi / P  # Mean motion
    M = n * t
    
    # Solve Kepler's equation to get eccentric anomaly
    E = solve_kepler(M, e)
    
    # Calculate true anomaly
    nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    
    # Calculate distance from central body
    r = a * (1 - e * np.cos(E))
    
    # Position in orbital plane
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    z_orb = 0
    
    # Rotate to account for inclination
    x = x_orb
    y = y_orb * np.cos(i) - z_orb * np.sin(i)
    z = y_orb * np.sin(i) + z_orb * np.cos(i)
    
    return np.array([x, y, z])

def get_velocity_at_time(orbit, t, mu=SUN_MU):
    """
    Calculate velocity of a body in its orbit at time t.
    
    Args:
        orbit: Dictionary with orbital elements
        t: Time in seconds
        mu: Gravitational parameter of central body
    
    Returns:
        Velocity vector [vx, vy, vz] in km/s
    """
    a = orbit['a']  # Semi-major axis
    e = orbit['e']  # Eccentricity
    i = orbit['i']  # Inclination
    P = orbit['P']  # Period
    
    # Calculate mean anomaly at time t
    n = 2 * np.pi / P  # Mean motion
    M = n * t
    
    # Solve Kepler's equation to get eccentric anomaly
    E = solve_kepler(M, e)
    
    # Calculate true anomaly
    nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    
    # Calculate distance and angular position
    r = a * (1 - e * np.cos(E))
    
    # Calculate velocity components in orbital plane
    p = a * (1 - e**2)  # Semi-latus rectum
    h = np.sqrt(mu * p)  # Angular momentum
    
    # Velocity in orbital plane
    vx_orb = -(h / r) * np.sin(nu)
    vy_orb = (h / r) * (e + np.cos(nu))
    vz_orb = 0
    
    # Rotate to account for inclination
    vx = vx_orb
    vy = vy_orb * np.cos(i) - vz_orb * np.sin(i)
    vz = vy_orb * np.sin(i) + vz_orb * np.cos(i)
    
    return np.array([vx, vy, vz])

def solve_kepler(M, e, tol=1e-10, max_iter=100):
    """
    Solve Kepler's equation for eccentric anomaly.
    
    Args:
        M: Mean anomaly (rad)
        e: Eccentricity
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
    
    Returns:
        Eccentric anomaly (rad)
    """
    # Ensure M is between 0 and 2*pi
    M = M % (2 * np.pi)
    
    # Initial guess
    if e < 0.8:
        E = M
    else:
        E = np.pi
        
    # Newton-Raphson iteration
    for i in range(max_iter):
        E_next = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        if abs(E_next - E) < tol:
            return E_next
        E = E_next
        
    return E

def calculate_hohmann_transfer(r1, r2, mu=SUN_MU):
    """
    Calculate parameters for a Hohmann transfer between circular orbits.
    
    Args:
        r1: Radius of departure orbit (km)
        r2: Radius of arrival orbit (km)
        mu: Gravitational parameter of central body
    
    Returns:
        Dictionary with transfer parameters
    """
    # Semi-major axis of transfer orbit
    a_transfer = (r1 + r2) / 2
    
    # Velocities in circular orbits
    v1_circ = np.sqrt(mu / r1)
    v2_circ = np.sqrt(mu / r2)
    
    # Velocities in transfer orbit at periapsis and apoapsis
    v1_transfer = np.sqrt(mu * (2 / r1 - 1 / a_transfer))
    v2_transfer = np.sqrt(mu * (2 / r2 - 1 / a_transfer))
    
    # Delta-Vs
    delta_v1 = abs(v1_transfer - v1_circ)
    delta_v2 = abs(v2_circ - v2_transfer)
    total_delta_v = delta_v1 + delta_v2
    
    # Time of flight
    tof = np.pi * np.sqrt(a_transfer**3 / mu)
    
    return {
        'a_transfer': a_transfer,
        'delta_v1': delta_v1,
        'delta_v2': delta_v2,
        'total_delta_v': total_delta_v,
        'tof': tof
    }

def propagate_orbit(r0, v0, times, mu=SUN_MU):
    """
    Propagate an orbit from initial position and velocity.
    
    Args:
        r0: Initial position vector [x, y, z] in km
        v0: Initial velocity vector [vx, vy, vz] in km/s
        times: Array of time points (seconds from start)
        mu: Gravitational parameter of central body
    
    Returns:
        Array of position vectors at each time point
    """
    # Calculate orbital elements from state vector
    r_mag = np.linalg.norm(r0)
    v_mag = np.linalg.norm(v0)
    
    # Specific energy
    energy = v_mag**2 / 2 - mu / r_mag
    
    # Semi-major axis
    a = -mu / (2 * energy)
    
    # Angular momentum vector
    h_vec = np.cross(r0, v0)
    h_mag = np.linalg.norm(h_vec)
    
    # Eccentricity vector
    evec = np.cross(v0, h_vec) / mu - r0 / r_mag
    e = np.linalg.norm(evec)
    
    # Node vector
    node_vec = np.cross([0, 0, 1], h_vec)
    node_mag = np.linalg.norm(node_vec)
    
    # Inclination
    inc = np.arccos(h_vec[2] / h_mag)
    
    # Right ascension of ascending node
    if node_mag < 1e-10:
        raan = 0
    else:
        raan = np.arccos(node_vec[0] / node_mag)
        if node_vec[1] < 0:
            raan = 2 * np.pi - raan
    
    # Argument of periapsis
    if node_mag < 1e-10:
        argp = np.arctan2(evec[1], evec[0])
    else:
        argp = np.arccos(np.dot(node_vec, evec) / (node_mag * e))
        if evec[2] < 0:
            argp = 2 * np.pi - argp
    
    # True anomaly at initial point
    if e < 1e-10:
        nu0 = np.arccos(r0[0] / r_mag)
        if r0[1] < 0:
            nu0 = 2 * np.pi - nu0
    else:
        cos_nu = np.dot(evec, r0) / (e * r_mag)
        if cos_nu > 1:
            cos_nu = 1
        elif cos_nu < -1:
            cos_nu = -1
        nu0 = np.arccos(cos_nu)
        if np.dot(r0, v0) < 0:
            nu0 = 2 * np.pi - nu0
    
    # Mean motion
    n = np.sqrt(mu / a**3)
    
    # Convert true anomaly to eccentric anomaly
    if e < 1e-10:
        E0 = nu0
    else:
        cos_E0 = (e + np.cos(nu0)) / (1 + e * np.cos(nu0))
        sin_E0 = np.sqrt(1 - e**2) * np.sin(nu0) / (1 + e * np.cos(nu0))
        E0 = np.arctan2(sin_E0, cos_E0)
    
    # Mean anomaly at initial time
    M0 = E0 - e * np.sin(E0)
    
    # Initialize trajectory array
    trajectory = np.zeros((len(times), 3))
    
    # Propagate orbit for each time point
    for i, t in enumerate(times):
        # Calculate mean anomaly at time t
        M = (M0 + n * t) % (2 * np.pi)
        
        # Solve Kepler's equation for eccentric anomaly
        E = solve_kepler(M, e)
        
        # Calculate true anomaly
        cos_nu = (np.cos(E) - e) / (1 - e * np.cos(E))
        sin_nu = np.sqrt(1 - e**2) * np.sin(E) / (1 - e * np.cos(E))
        nu = np.arctan2(sin_nu, cos_nu)
        
        # Calculate radius
        r = a * (1 - e * np.cos(E))
        
        # Position in orbital plane
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        z_orb = 0
        
        # Rotation matrices
        R3_W = np.array([
            [np.cos(raan), -np.sin(raan), 0],
            [np.sin(raan), np.cos(raan), 0],
            [0, 0, 1]
        ])
        
        R1_i = np.array([
            [1, 0, 0],
            [0, np.cos(inc), -np.sin(inc)],
            [0, np.sin(inc), np.cos(inc)]
        ])
        
        R3_w = np.array([
            [np.cos(argp), -np.sin(argp), 0],
            [np.sin(argp), np.cos(argp), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix
        R = np.matmul(R3_W, np.matmul(R1_i, R3_w))
        
        # Rotate position to inertial frame
        pos_orb = np.array([x_orb, y_orb, z_orb])
        trajectory[i] = np.matmul(R, pos_orb)
    
    return trajectory

def calculate_earth_mars_transfer(departure_date, tof_days):
    """
    Calculate an Earth-to-Mars transfer trajectory.
    
    Args:
        departure_date: Departure date (YYYY-MM-DD)
        tof_days: Time of flight in days
    
    Returns:
        Dictionary with trajectory data
    """
    # Convert time of flight to seconds
    tof = tof_days * DAY
    
    # Convert departure date to seconds from J2000
    dt_format = "%Y-%m-%d"
    j2000 = datetime.datetime.strptime("2000-01-01", dt_format)
    departure = datetime.datetime.strptime(departure_date, dt_format)
    
    departure_seconds = (departure - j2000).total_seconds()
    arrival_seconds = departure_seconds + tof
    
    # Calculate positions at departure and arrival
    earth_pos_dep = get_position_at_time(EARTH_ORBIT, departure_seconds)
    earth_vel_dep = get_velocity_at_time(EARTH_ORBIT, departure_seconds)
    mars_pos_arr = get_position_at_time(MARS_ORBIT, arrival_seconds)
    mars_vel_arr = get_velocity_at_time(MARS_ORBIT, arrival_seconds)
    
    print(f"Earth position at departure: {earth_pos_dep} km")
    print(f"Earth velocity at departure: {earth_vel_dep} km/s")
    print(f"Mars position at arrival: {mars_pos_arr} km")
    print(f"Mars velocity at arrival: {mars_vel_arr} km/s")
    
    # Calculate transfer orbit parameters
    # Using patched conics approach with Lambert solver
    
    # Approximate the transfer as 180° Hohmann transfer for initial guess
    earth_sun_dist = np.linalg.norm(earth_pos_dep)
    mars_sun_dist = np.linalg.norm(mars_pos_arr)
    hohmann = calculate_hohmann_transfer(earth_sun_dist, mars_sun_dist)
    
    print(f"Hohmann transfer parameters:")
    print(f"  Semi-major axis: {hohmann['a_transfer'] / AU:.3f} AU")
    print(f"  Earth departure ΔV: {hohmann['delta_v1']:.2f} km/s")
    print(f"  Mars arrival ΔV: {hohmann['delta_v2']:.2f} km/s")
    print(f"  Total ΔV: {hohmann['total_delta_v']:.2f} km/s")
    print(f"  Time of flight: {hohmann['tof'] / DAY:.1f} days")
    
    # Adjust the velocity to place spacecraft on transfer trajectory
    # For a simple model, we'll use the velocity from the Hohmann transfer
    a_transfer = hohmann['a_transfer']
    
    # Departure velocity (heliocentric)
    earth_v_mag = np.linalg.norm(earth_vel_dep)
    departure_v_mag = np.sqrt(SUN_MU * (2 / earth_sun_dist - 1 / a_transfer))
    
    # Calculate transfer heading
    earth_to_mars = mars_pos_arr - earth_pos_dep
    transfer_angle = np.arccos(np.dot(earth_pos_dep, earth_to_mars) / 
                             (earth_sun_dist * np.linalg.norm(earth_to_mars)))
    
    # Create a velocity vector perpendicular to position in the orbital plane
    h_unit = np.array([0, 0, 1])  # Simplified assumption for orbital plane
    earth_dep_dir = earth_pos_dep / earth_sun_dist
    
    # Direction of departure velocity
    v_dep_dir = np.cross(h_unit, earth_dep_dir)
    v_dep_dir = v_dep_dir / np.linalg.norm(v_dep_dir)
    
    v_sc_dep = departure_v_mag * v_dep_dir
    
    # Calculate delta-v at departure
    delta_v_dep = np.linalg.norm(v_sc_dep - earth_vel_dep)
    
    # Arrival velocity (heliocentric)
    mars_v_mag = np.linalg.norm(mars_vel_arr)
    arrival_v_mag = np.sqrt(SUN_MU * (2 / mars_sun_dist - 1 / a_transfer))
    
    # Direction of arrival velocity
    mars_arr_dir = mars_pos_arr / mars_sun_dist
    v_arr_dir = np.cross(h_unit, mars_arr_dir)
    v_arr_dir = v_arr_dir / np.linalg.norm(v_arr_dir)
    
    v_sc_arr = arrival_v_mag * v_arr_dir
    
    # Calculate delta-v at arrival
    delta_v_arr = np.linalg.norm(v_sc_arr - mars_vel_arr)
    
    print(f"\nCalculated transfer parameters:")
    print(f"  Earth departure date: {departure_date}")
    print(f"  Mars arrival date: {(departure + datetime.timedelta(days=tof_days)).strftime('%Y-%m-%d')}")
    print(f"  Time of flight: {tof_days:.1f} days")
    print(f"  Earth departure ΔV: {delta_v_dep:.2f} km/s")
    print(f"  Mars arrival ΔV: {delta_v_arr:.2f} km/s")
    print(f"  Total ΔV: {delta_v_dep + delta_v_arr:.2f} km/s")
    
    # Generate trajectories
    n_points = 500
    
    # Spacecraft trajectory
    times = np.linspace(0, tof, n_points)
    sc_trajectory = propagate_orbit(earth_pos_dep, v_sc_dep, times)
    
    # Earth and Mars trajectories over the transfer period
    earth_trajectory = np.zeros((n_points, 3))
    mars_trajectory = np.zeros((n_points, 3))
    
    for i, t in enumerate(times):
        time_sec = departure_seconds + t
        earth_trajectory[i] = get_position_at_time(EARTH_ORBIT, time_sec)
        mars_trajectory[i] = get_position_at_time(MARS_ORBIT, time_sec)
    
    # Complete mission info
    transfer_data = {
        'departure_date': departure_date,
        'arrival_date': (departure + datetime.timedelta(days=tof_days)).strftime('%Y-%m-%d'),
        'tof_days': tof_days,
        'delta_v_dep': delta_v_dep,
        'delta_v_arr': delta_v_arr,
        'earth_pos_dep': earth_pos_dep,
        'earth_vel_dep': earth_vel_dep,
        'v_sc_dep': v_sc_dep,
        'mars_pos_arr': mars_pos_arr,
        'mars_vel_arr': mars_vel_arr,
        'v_sc_arr': v_sc_arr,
        'sc_trajectory': sc_trajectory,
        'earth_trajectory': earth_trajectory,
        'mars_trajectory': mars_trajectory
    }
    
    return transfer_data

def calculate_mission_parameters(transfer_data):
    """
    Calculate detailed mission parameters based on the transfer trajectory.
    
    Args:
        transfer_data: Dictionary with transfer trajectory data
    
    Returns:
        Dictionary with mission parameters
    """
    # Extract basic parameters
    delta_v_dep = transfer_data['delta_v_dep']
    delta_v_arr = transfer_data['delta_v_arr']
    earth_pos_dep = transfer_data['earth_pos_dep']
    v_sc_dep = transfer_data['v_sc_dep']
    
    # Earth parameters
    earth_radius = 6378.0  # km
    parking_alt = 300.0  # km
    parking_radius = earth_radius + parking_alt
    
    # Earth parking orbit velocity
    parking_velocity = np.sqrt(EARTH_MU / parking_radius)  # km/s
    
    # Trans-Mars injection (TMI) burn calculation
    v_inf_earth = delta_v_dep  # Simplification for this model
    tmi_delta_v = np.sqrt(v_inf_earth**2 + 2 * EARTH_MU / parking_radius) - parking_velocity
    
    # Mars parameters
    mars_radius = 3396.0  # km
    mars_orbit_alt = 400.0  # km
    mars_orbit_radius = mars_radius + mars_orbit_alt
    
    # Mars orbit insertion (MOI) burn calculation
    mars_orbit_velocity = np.sqrt(MARS_MU / mars_orbit_radius)  # km/s
    v_inf_mars = delta_v_arr  # Simplification for this model
    moi_delta_v = np.sqrt(v_inf_mars**2 + 2 * MARS_MU / mars_orbit_radius) - mars_orbit_velocity
    
    # Trajectory characteristics
    sc_trajectory = transfer_data['sc_trajectory']
    min_sun_dist = min(np.linalg.norm(pos) for pos in sc_trajectory)
    max_sun_dist = max(np.linalg.norm(pos) for pos in sc_trajectory)
    
    # Semi-major axis and eccentricity
    a = (min_sun_dist + max_sun_dist) / 2
    e = (max_sun_dist - min_sun_dist) / (max_sun_dist + min_sun_dist)
    
    # C3 (characteristic energy)
    c3 = np.linalg.norm(v_sc_dep)**2 - 2 * EARTH_MU / np.linalg.norm(earth_pos_dep)
    
    mission_params = {
        'tmi_delta_v': tmi_delta_v,
        'moi_delta_v': moi_delta_v,
        'total_mission_delta_v': tmi_delta_v + moi_delta_v,
        'transfer_a': a,
        'transfer_e': e,
        'min_sun_dist': min_sun_dist,
        'max_sun_dist': max_sun_dist,
        'c3': c3,
        'parking_velocity': parking_velocity,
        'mars_orbit_velocity': mars_orbit_velocity
    }
    
    # Print mission analysis
    print("\n===== MISSION PARAMETER ANALYSIS =====")
    
    print("\nTransfer Orbit Characteristics:")
    print(f"  Semi-major axis: {a:.2f} km ({a/AU:.3f} AU)")
    print(f"  Eccentricity: {e:.4f}")
    print(f"  Perihelion distance: {min_sun_dist:.2f} km ({min_sun_dist/AU:.3f} AU)")
    print(f"  Aphelion distance: {max_sun_dist:.2f} km ({max_sun_dist/AU:.3f} AU)")
    
    print("\nEarth Departure:")
    print(f"  Earth parking orbit altitude: {parking_alt:.1f} km")
    print(f"  Parking orbit velocity: {parking_velocity:.2f} km/s")
    print(f"  Hyperbolic excess velocity: {v_inf_earth:.2f} km/s")
    print(f"  C3 (characteristic energy): {c3:.2f} km²/s²")
    print(f"  TMI Delta-V (from parking orbit): {tmi_delta_v:.2f} km/s")
    
    print("\nMars Arrival:")
    print(f"  Mars orbit altitude: {mars_orbit_alt:.1f} km")
    print(f"  Mars orbit velocity: {mars_orbit_velocity:.2f} km/s")
    print(f"  Hyperbolic approach velocity: {v_inf_mars:.2f} km/s")
    print(f"  MOI Delta-V (for capture): {moi_delta_v:.2f} km/s")
    
    print("\nTotal Mission Delta-V:")
    print(f"  TMI (Earth departure): {tmi_delta_v:.2f} km/s")
    print(f"  MOI (Mars capture): {moi_delta_v:.2f} km/s")
    print(f"  Total required: {tmi_delta_v + moi_delta_v:.2f} km/s")
    
    return mission_params

def plot_transfer_trajectory(transfer_data, output_file="earth_mars_transfer.png"):
    """
    Plot the Earth-to-Mars transfer trajectory.
    
    Args:
        transfer_data: Dictionary with transfer trajectory data
        output_file: File to save the plot
    """
    print("Generating transfer trajectory plot...")
    
    # Extract data
    departure_date = transfer_data['departure_date']
    arrival_date = transfer_data['arrival_date']
    tof_days = transfer_data['tof_days']
    delta_v_dep = transfer_data['delta_v_dep']
    delta_v_arr = transfer_data['delta_v_arr']
    
    sc_trajectory = transfer_data['sc_trajectory']
    earth_trajectory = transfer_data['earth_trajectory']
    mars_trajectory = transfer_data['mars_trajectory']
    
    earth_pos_dep = transfer_data['earth_pos_dep']
    mars_pos_arr = transfer_data['mars_pos_arr']
    
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
    
    # Plot planets
    ax.scatter(0, 0, 0, color='gold', s=200, edgecolor='black', label='Sun')
    ax.scatter(earth_pos_dep[0], earth_pos_dep[1], earth_pos_dep[2], 
              color='royalblue', s=100, edgecolor='black', label='Earth at Departure')
    ax.scatter(mars_pos_arr[0], mars_pos_arr[1], mars_pos_arr[2], 
              color='firebrick', s=80, edgecolor='black', label='Mars at Arrival')
    
    # Mark transfer trajectory
    ax.scatter(sc_trajectory[0, 0], sc_trajectory[0, 1], sc_trajectory[0, 2], 
              color='green', s=80, marker='^', edgecolor='black', label='Departure')
    ax.scatter(sc_trajectory[-1, 0], sc_trajectory[-1, 1], sc_trajectory[-1, 2], 
              color='green', s=80, marker='v', edgecolor='black', label='Arrival')
    
    # Draw radial lines to planets
    ax.plot([0, earth_pos_dep[0]], [0, earth_pos_dep[1]], [0, earth_pos_dep[2]], 
           'k:', alpha=0.3)
    ax.plot([0, mars_pos_arr[0]], [0, mars_pos_arr[1]], [0, mars_pos_arr[2]], 
           'k:', alpha=0.3)
    
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
    ax.set_title('Earth-to-Mars Transfer Trajectory')
    
    # Add mission info
    mission_info = (
        f"Mission Parameters:\n"
        f"Departure: {departure_date}\n"
        f"Arrival: {arrival_date}\n"
        f"Time of Flight: {tof_days} days\n"
        f"Earth Departure ΔV: {delta_v_dep:.2f} km/s\n"
        f"Mars Arrival ΔV: {delta_v_arr:.2f} km/s\n"
        f"Total ΔV: {delta_v_dep + delta_v_arr:.2f} km/s"
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

def main():
    """Main function to run the Earth-to-Mars mission simulation."""
    try:
        print("Earth-to-Mars Transfer Trajectory Simulation")
        print("===========================================")
        
        # Mission parameters for a typical Hohmann-like transfer
        departure_date = "2022-05-15"  # Selected for favorable alignment
        tof_days = 210  # Typical Earth-to-Mars transfer time (~7 months)
        
        print(f"Calculating transfer from Earth to Mars")
        print(f"Departure date: {departure_date}")
        print(f"Time of flight: {tof_days} days")
        
        # Calculate transfer trajectory
        transfer_data = calculate_earth_mars_transfer(departure_date, tof_days)
        
        # Analyze mission parameters
        mission_params = calculate_mission_parameters(transfer_data)
        
        # Create visualization
        plot_transfer_trajectory(transfer_data, "earth_mars_transfer.png")
        
        print("\nEarth-to-Mars mission simulation completed successfully!")
        
    except Exception as e:
        print(f"Error in mission simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 