#!/usr/bin/env python3
"""
Earth-to-Mars Interplanetary Spacecraft Mission Simulator

This script designs and visualizes an optimal trajectory for a spacecraft mission
from Earth to Mars, including launch window analysis, transfer orbit calculation,
and mission parameter optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import spiceypy as spice
import os
from datetime import datetime, timedelta
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors

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

def search_launch_window(start_date, end_date, step_days, tof_min_days, tof_max_days, tof_step_days):
    """
    Search for optimal launch windows for Earth-to-Mars transfers.
    
    Args:
        start_date: Start date for launch window search (ISO format)
        end_date: End date for launch window search (ISO format)
        step_days: Step size in days for the search
        tof_min_days: Minimum time of flight in days
        tof_max_days: Maximum time of flight in days
        tof_step_days: Step size for time of flight in days
        
    Returns:
        Dictionary with launch window information
    """
    print(f"Searching for launch windows from {start_date} to {end_date}...")
    
    # Convert dates to ephemeris time
    start_et = spice.str2et(start_date)
    end_et = spice.str2et(end_date)
    
    # Constants
    earth_id = 399  # SPICE ID for Earth
    mars_id = 499   # SPICE ID for Mars
    sun_id = 10     # SPICE ID for Sun
    seconds_per_day = 86400
    
    # Arrays to store results
    launch_dates = []
    arrival_dates = []
    delta_vs = []
    tofs = []
    c3s = []  # Characteristic energy (C3)
    
    # Track search progress
    total_steps = int((end_et - start_et) / (step_days * seconds_per_day)) + 1
    count = 0
    found_count = 0
    
    # Loop through potential launch dates
    current_et = start_et
    while current_et <= end_et:
        count += 1
        if count % 10 == 0:
            print(f"Progress: {count}/{total_steps} dates examined ({found_count} valid trajectories found)")
            
        current_date = spice.et2utc(current_et, "ISOC", 0)
        
        # Get Earth state at departure
        earth_state = get_body_state_at_time(earth_id, current_et)
        r1 = earth_state[:3]  # Earth position at departure
        v1_earth = earth_state[3:]  # Earth velocity at departure
        
        # Try different transfer times
        for tof_days in np.arange(tof_min_days, tof_max_days + 1, tof_step_days):
            tof_seconds = tof_days * seconds_per_day
            arrival_et = current_et + tof_seconds
            arrival_date = spice.et2utc(arrival_et, "ISOC", 0)
            
            try:
                # Get Mars state at arrival
                mars_state = get_body_state_at_time(mars_id, arrival_et)
                r2 = mars_state[:3]  # Mars position at arrival
                v2_mars = mars_state[3:]  # Mars velocity at arrival
                
                # Solve Lambert's problem to find transfer velocities
                v1, v2 = lambert_problem(r1, r2, tof_seconds, mu=SUN_MU)
                
                # Calculate delta-v at departure and arrival
                delta_v1 = np.linalg.norm(v1 - v1_earth)
                delta_v2 = np.linalg.norm(v2 - v2_mars)
                total_delta_v = delta_v1 + delta_v2
                
                # Calculate C3 (characteristic energy)
                c3 = np.linalg.norm(v1)**2 - 2*EARTH_MU/np.linalg.norm(r1)
                
                # Store result
                launch_dates.append(current_date)
                arrival_dates.append(arrival_date)
                delta_vs.append(total_delta_v)
                tofs.append(tof_days)
                c3s.append(c3)
                found_count += 1
                
                if found_count % 50 == 0:
                    print(f"Found {found_count} valid trajectories")
                
            except Exception as e:
                # Skip problematic cases
                pass
        
        # Advance to next potential launch date
        current_et += step_days * seconds_per_day
    
    # Organize results
    results = {
        'launch_dates': launch_dates,
        'arrival_dates': arrival_dates,
        'delta_vs': delta_vs,
        'tofs': tofs,
        'c3s': c3s
    }
    
    print(f"Search complete. Found {found_count} valid trajectories.")
    
    return results

def plot_porkchop(results, output_file="porkchop_plot.png"):
    """
    Create a porkchop plot for Earth-to-Mars transfers.
    
    Args:
        results: Dictionary with launch window information
        output_file: File to save the plot
    """
    print("Generating porkchop plot...")
    
    # Check if we have any results
    if not results['launch_dates']:
        print("No valid trajectories found. Cannot create porkchop plot.")
        # Return a default value for optimal launch
        default_departure = "2022-09-20T00:00:00"
        default_arrival = "2023-05-10T00:00:00" 
        default_info = {
            'launch_date': default_departure,
            'arrival_date': default_arrival,
            'tof_days': 230,
            'delta_v': 5.5,
            'c3': 10.0,
            'launch_et': spice.str2et(default_departure)
        }
        print(f"Using default Earth-Mars trajectory: {default_departure} to {default_arrival}")
        return default_info
    
    # Convert dates to numerical values for plotting
    launch_dates_num = [spice.str2et(date) for date in results['launch_dates']]
    tof_days = results['tofs']
    delta_vs = results['delta_vs']
    
    # Create a 2D grid for contour plotting
    launch_dates_unique = sorted(list(set(launch_dates_num)))
    tof_unique = sorted(list(set(tof_days)))
    
    print(f"Unique launch dates: {len(launch_dates_unique)}")
    print(f"Unique TOF values: {len(tof_unique)}")
    
    # Initialize grid
    delta_v_grid = np.ones((len(tof_unique), len(launch_dates_unique))) * np.nan
    
    # Fill the grid
    for i, (launch_date, tof, delta_v) in enumerate(zip(launch_dates_num, tof_days, delta_vs)):
        x_idx = launch_dates_unique.index(launch_date)
        y_idx = tof_unique.index(tof)
        delta_v_grid[y_idx, x_idx] = delta_v
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create contour plot with proper levels
    min_delta_v = np.nanmin(delta_vs)
    max_delta_v = np.nanmax(delta_vs)
    print(f"Delta-V range: {min_delta_v:.2f} to {max_delta_v:.2f} km/s")
    
    levels = np.linspace(min_delta_v, min(max_delta_v, min_delta_v * 2), 20)
    cs = ax.contourf(launch_dates_unique, tof_unique, delta_v_grid, levels=levels, cmap='viridis')
    
    # Add contour lines with labels
    contour_levels = np.linspace(min_delta_v, min(max_delta_v, min_delta_v * 1.5), 10)
    cs2 = ax.contour(launch_dates_unique, tof_unique, delta_v_grid, levels=contour_levels, colors='white', linestyles='dashed', linewidths=0.5)
    ax.clabel(cs2, fmt='%.1f km/s', fontsize=8, colors='white')
    
    # Find minimum delta-v
    min_idx = np.nanargmin(delta_vs)
    min_date = results['launch_dates'][min_idx]
    min_tof = results['tofs'][min_idx]
    min_delta_v = delta_vs[min_idx]
    min_date_num = launch_dates_num[min_idx]
    
    print(f"Optimal launch: {min_date}, TOF: {min_tof} days, Delta-V: {min_delta_v:.2f} km/s")
    
    # Mark minimum point
    ax.scatter(min_date_num, min_tof, color='red', s=50, edgecolor='white', zorder=5)
    ax.annotate(f"Min ΔV: {min_delta_v:.2f} km/s\nLaunch: {min_date}\nTOF: {min_tof} days",
                xy=(min_date_num, min_tof), xytext=(min_date_num, min_tof + 20),
                arrowprops=dict(arrowstyle="->", color='white'), color='white',
                bbox=dict(boxstyle="round,pad=0.3", fc='black', alpha=0.7))
    
    # Format x-axis (launch dates)
    date_formatter = plt.FuncFormatter(lambda x, pos: spice.et2utc(x, "ISOC", 0)[:10])
    ax.xaxis.set_major_formatter(date_formatter)
    fig.autofmt_xdate(rotation=45)
    
    # Labels and title
    ax.set_xlabel('Launch Date')
    ax.set_ylabel('Time of Flight (days)')
    ax.set_title('Earth-to-Mars Transfer Porkchop Plot\nTotal ΔV (km/s)')
    
    # Colorbar
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label('Total ΔV (km/s)')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Porkchop plot saved to {output_file}")
    
    # Return minimum values
    min_info = {
        'launch_date': min_date,
        'arrival_date': results['arrival_dates'][min_idx],
        'tof_days': min_tof,
        'delta_v': min_delta_v,
        'c3': results['c3s'][min_idx],
        'launch_et': min_date_num
    }
    
    return min_info

def simulate_transfer_trajectory(departure_date, arrival_date):
    """
    Simulate an Earth-to-Mars transfer trajectory.
    
    Args:
        departure_date: Departure date (ISO format)
        arrival_date: Arrival date (ISO format)
        
    Returns:
        Dictionary with trajectory information
    """
    print(f"Simulating transfer trajectory from Earth on {departure_date} to Mars on {arrival_date}...")
    
    # Convert dates to ephemeris time
    departure_et = spice.str2et(departure_date)
    arrival_et = spice.str2et(arrival_date)
    tof = arrival_et - departure_et
    
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
    v1_trans, v2_trans = lambert_problem(r1, r2, tof, mu=SUN_MU)
    
    # Calculate delta-Vs
    delta_v1 = np.linalg.norm(v1_trans - v1_earth)
    delta_v2 = np.linalg.norm(v2_trans - v2_mars)
    total_delta_v = delta_v1 + delta_v2
    
    # Calculate C3 (characteristic energy)
    c3 = np.linalg.norm(v1_trans)**2 - 2*EARTH_MU/np.linalg.norm(r1)
    
    print(f"Transfer Orbit Parameters:")
    print(f"  Departure: {departure_date}")
    print(f"  Arrival: {arrival_date}")
    print(f"  Time of Flight: {tof/86400:.1f} days")
    print(f"  Earth Departure ΔV: {delta_v1:.2f} km/s")
    print(f"  Mars Arrival ΔV: {delta_v2:.2f} km/s")
    print(f"  Total ΔV: {total_delta_v:.2f} km/s")
    print(f"  C3: {c3:.2f} km²/s²")
    
    # Propagate spacecraft's trajectory
    n_points = 500
    traj_times = np.linspace(0, tof, n_points)
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
        'tof_seconds': tof,
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
        f"Time of Flight: {trajectory_info['tof_seconds']/86400:.1f} days\n"
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

def create_trajectory_animation(trajectory_info, output_file="earth_mars_transfer.mp4"):
    """
    Create an animation of the Earth-to-Mars transfer.
    
    Args:
        trajectory_info: Dictionary with trajectory information
        output_file: File to save the animation
    """
    print("Creating animation of the transfer trajectory...")
    
    # Extract data
    sc_trajectory = trajectory_info['sc_trajectory']
    earth_trajectory = trajectory_info['earth_trajectory']
    mars_trajectory = trajectory_info['mars_trajectory']
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set equal aspect ratio
    max_val = max(
        np.max(np.abs(earth_trajectory)),
        np.max(np.abs(mars_trajectory)),
        np.max(np.abs(sc_trajectory))
    )
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_zlim(-max_val, max_val)
    
    # Initialize plot elements
    sun = ax.scatter([0], [0], [0], color=PLANET_COLORS['Sun'], s=200, edgecolor='black', label='Sun')
    earth_orbit, = ax.plot([], [], [], 'b-', label='Earth Orbit', linewidth=1.5, alpha=0.7)
    mars_orbit, = ax.plot([], [], [], 'r-', label='Mars Orbit', linewidth=1.5, alpha=0.7)
    sc_traj, = ax.plot([], [], [], 'g--', label='Spacecraft Trajectory', linewidth=1.5)
    earth = ax.scatter([], [], [], color=PLANET_COLORS['Earth'], s=100, edgecolor='black', label='Earth')
    mars = ax.scatter([], [], [], color=PLANET_COLORS['Mars'], s=80, edgecolor='black', label='Mars')
    spacecraft = ax.scatter([], [], [], color='lime', s=50, edgecolor='black', marker='D', label='Spacecraft')
    
    # Add a time counter
    time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    
    # Labels and title
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Earth-to-Mars Interplanetary Transfer Animation')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Animation update function
    def update(frame):
        # Update orbit trails (growing as the animation progresses)
        frame_idx = min(frame + 1, len(earth_trajectory))
        
        earth_orbit.set_data(earth_trajectory[:frame_idx, 0], earth_trajectory[:frame_idx, 1])
        earth_orbit.set_3d_properties(earth_trajectory[:frame_idx, 2])
        
        mars_orbit.set_data(mars_trajectory[:frame_idx, 0], mars_trajectory[:frame_idx, 1])
        mars_orbit.set_3d_properties(mars_trajectory[:frame_idx, 2])
        
        sc_traj.set_data(sc_trajectory[:frame_idx, 0], sc_trajectory[:frame_idx, 1])
        sc_traj.set_3d_properties(sc_trajectory[:frame_idx, 2])
        
        # Update current positions
        earth._offsets3d = ([earth_trajectory[frame, 0]], [earth_trajectory[frame, 1]], [earth_trajectory[frame, 2]])
        mars._offsets3d = ([mars_trajectory[frame, 0]], [mars_trajectory[frame, 1]], [mars_trajectory[frame, 2]])
        spacecraft._offsets3d = ([sc_trajectory[frame, 0]], [sc_trajectory[frame, 1]], [sc_trajectory[frame, 2]])
        
        # Update time counter (days since departure)
        tof_days = trajectory_info['tof_seconds'] / 86400
        current_day = frame * tof_days / len(sc_trajectory)
        time_text.set_text(f'Time: {current_day:.1f} days')
        
        # Slowly rotate view
        ax.view_init(elev=30, azim=frame/5)
        
        return earth_orbit, mars_orbit, sc_traj, earth, mars, spacecraft, time_text
    
    # Create animation with 200 frames
    n_frames = 200
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)
    
    # Save animation
    ani.save(output_file, writer='ffmpeg', fps=30, dpi=200)
    print(f"Animation saved to {output_file}")

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
    tof_days = trajectory_info['tof_seconds'] / 86400
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
    print(f"  Time of Flight: {tof_days:.1f} days")
    
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
        
        # Define search parameters
        start_date = "2022-01-01T00:00:00"
        end_date = "2023-01-01T00:00:00"  # Reduced search range to make it faster
        step_days = 30                     # Increased step size to reduce computation
        tof_min_days = 180
        tof_max_days = 300
        tof_step_days = 20                 # Increased step size to reduce computation
        
        # Search for launch windows
        launch_windows = search_launch_window(
            start_date, end_date, step_days, 
            tof_min_days, tof_max_days, tof_step_days
        )
        
        # Create porkchop plot and get optimal launch info
        optimal_launch = plot_porkchop(launch_windows, "porkchop_plot.png")
        
        # Simulate optimal transfer trajectory
        trajectory = simulate_transfer_trajectory(
            optimal_launch['launch_date'],
            optimal_launch['arrival_date']
        )
        
        # Analyze mission parameters
        mission_analysis = analyze_mission_parameters(trajectory)
        
        # Create visualization
        plot_transfer_trajectory(trajectory, "earth_mars_transfer.png")
        
        # Create animation (optional - may require ffmpeg)
        try:
            create_trajectory_animation(trajectory, "earth_mars_transfer.mp4")
        except Exception as e:
            print(f"Could not create animation: {e}")
            print("Animation requires ffmpeg to be installed.")
        
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