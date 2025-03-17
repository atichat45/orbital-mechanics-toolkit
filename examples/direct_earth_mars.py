#!/usr/bin/env python3
"""
Direct Earth-to-Mars Interplanetary Trajectory Simulator

This script calculates an Earth-to-Mars transfer orbit for a specific launch date
and visualizes the trajectory.
"""

import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
import os
from datetime import datetime, timedelta
import math
from mpl_toolkits.mplot3d import Axes3D

from orbital_sim.models.two_body import propagate_trajectory
from orbital_sim.models.orbital_mechanics import lambert_problem

# Standard gravitational parameters (GM) in km^3/s^2
SUN_MU = 1.32712440018e20
EARTH_MU = 3.986004418e14
MARS_MU = 4.282837e13

# Plotting constants
PLANET_COLORS = {
    'Sun': 'gold',
    'Earth': 'royalblue',
    'Mars': 'firebrick'
}

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

def get_body_state_at_time(body_id, epoch, frame="ECLIPJ2000", observer_id=10):
    """
    Get the state vector (position and velocity) of a celestial body at a given time.
    
    Args:
        body_id: SPICE ID of the target body
        epoch: Epoch time (string in ISO format or TDB seconds past J2000)
        frame: Reference frame
        observer_id: SPICE ID of the observer body (default: Sun)
        
    Returns:
        State vector [x, y, z, vx, vy, vz] in km and km/s
    """
    # Convert epoch to ET if it's a string
    if isinstance(epoch, str):
        et = spice.str2et(epoch)
    else:
        et = epoch
        
    # Get the state vector
    state, lt = spice.spkezr(
        str(body_id), 
        et, 
        frame, 
        "NONE", 
        str(observer_id)
    )
    
    return np.array(state)

def simulate_transfer_trajectory(departure_date, tof_days):
    """
    Simulate an Earth-to-Mars transfer trajectory.
    
    Args:
        departure_date: Departure date (ISO format)
        tof_days: Time of flight in days
        
    Returns:
        Dictionary with trajectory information
    """
    # Convert dates to ephemeris time
    departure_et = spice.str2et(departure_date)
    seconds_per_day = 86400
    tof_seconds = tof_days * seconds_per_day
    arrival_et = departure_et + tof_seconds
    arrival_date = spice.et2utc(arrival_et, "ISOC", 0)
    
    print(f"Simulating transfer trajectory from Earth on {departure_date} to Mars on {arrival_date}...")
    print(f"Time of flight: {tof_days} days")
    
    # Get body IDs
    earth_id = 399  # SPICE ID for Earth
    mars_id = 499   # SPICE ID for Mars
    sun_id = 10     # SPICE ID for Sun
    
    # Get states at departure and arrival
    earth_state_dep = get_body_state_at_time(earth_id, departure_et)
    mars_state_arr = get_body_state_at_time(mars_id, arrival_et)
    
    # Extract positions and velocities
    r1 = earth_state_dep[:3]  # Earth position at departure
    v1_earth = earth_state_dep[3:]  # Earth velocity at departure
    r2 = mars_state_arr[:3]  # Mars position at arrival
    v2_mars = mars_state_arr[3:]  # Mars velocity at arrival
    
    # Solve Lambert's problem for transfer orbit
    v1_trans, v2_trans = lambert_problem(r1, r2, tof_seconds, mu=SUN_MU)
    
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
    sc_trajectory = propagate_trajectory(r1, v1_trans, traj_times, SUN_MU)
    
    # Get Earth and Mars trajectories over the transfer period
    earth_trajectory = np.zeros((n_points, 3))
    mars_trajectory = np.zeros((n_points, 3))
    
    for i, t in enumerate(np.linspace(departure_et, arrival_et, n_points)):
        earth_state = get_body_state_at_time(earth_id, t)
        mars_state = get_body_state_at_time(mars_id, t)
        earth_trajectory[i] = earth_state[:3]
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
        'times': np.linspace(departure_et, arrival_et, n_points)
    }
    
    return trajectory_info

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
    ax.scatter(0, 0, 0, color=PLANET_COLORS['Sun'], s=300, edgecolor='black', label='Sun')
    ax.scatter(r1[0], r1[1], r1[2], color=PLANET_COLORS['Earth'], s=150, edgecolor='black', label='Earth at Departure')
    ax.scatter(r2[0], r2[1], r2[2], color=PLANET_COLORS['Mars'], s=100, edgecolor='black', label='Mars at Arrival')
    
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
        
        # Set fixed mission parameters
        departure_date = "2022-01-20T00:00:00"  # January 2022
        tof_days = 210  # ~7 months
        
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