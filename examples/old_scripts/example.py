#!/usr/bin/env python3
"""
Example script demonstrating a simple orbital mechanics simulation.
"""

import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
import os
from datetime import datetime, timedelta
import math

from orbital_sim.models.two_body import propagate_trajectory

# Standard gravitational parameter for the Sun (GM) in km^3/s^2
SUN_MU = 1.32712440018e20

def simple_earth_orbit():
    """
    Simulate Earth's orbit around the Sun using two-body propagation.
    
    This function:
    1. Loads SPICE kernels directly
    2. Retrieves Earth's state vector
    3. Propagates Earth's orbit for one year
    4. Visualizes the orbit
    """
    print("Running simple Earth orbit example...")
    
    # Load kernels directly
    kernels_dir = 'data/kernels'
    lsk_path = os.path.join(kernels_dir, 'naif0012.tls')
    spk_path = os.path.join(kernels_dir, 'de440.bsp')
    pck_path = os.path.join(kernels_dir, 'pck00010.tpc')
    
    print("Loading SPICE kernels...")
    spice.furnsh(lsk_path)
    spice.furnsh(spk_path)
    spice.furnsh(pck_path)
    
    try:
        # Define start time
        start_date = "2000-01-01T00:00:00"
        start_et = spice.str2et(start_date)
        
        # Simulation for one Earth year
        days_in_year = 365.25
        seconds_in_day = 86400
        duration = days_in_year * seconds_in_day
        
        # IDs for bodies
        earth_id = 399  # SPICE ID for Earth
        sun_id = 10     # SPICE ID for Sun
        
        print("Getting Earth's initial state...")
        earth_state, _ = spice.spkezr(str(earth_id), start_et, "ECLIPJ2000", "NONE", str(sun_id))
        
        # Extract position and velocity
        r0 = np.array(earth_state[:3])  # Earth position at start
        v0 = np.array(earth_state[3:])  # Earth velocity at start
        
        print("Earth position (km):", r0)
        print("Earth velocity (km/s):", v0)
        print("Position magnitude (km):", np.linalg.norm(r0))
        print("Velocity magnitude (km/s):", np.linalg.norm(v0))
        
        # Using hardcoded Sun's gravitational parameter
        mu = SUN_MU
        print(f"Sun's gravitational parameter: {mu} km^3/s^2")
        
        # Known Earth orbital parameters (average values)
        semi_major_axis = 149.6e6  # km
        eccentricity = 0.0167
        inclination = 0.0  # deg (to ecliptic)
        period_days = 365.25  # days
        
        print("\nEarth's Standard Orbital Elements:")
        print(f"Semi-major axis: {semi_major_axis:.2f} km")
        print(f"Eccentricity: {eccentricity:.6f}")
        print(f"Inclination: {inclination:.2f} deg")
        print(f"Period: {period_days:.2f} days")
        
        # Set up time points for propagation
        n_points = 500
        times = np.linspace(0, duration, n_points)
        
        print("\nPropagating Earth's orbit using SPICE state vectors and numerical integration...")
        earth_traj = propagate_trajectory(r0, v0, times, mu)
        
        # Track the Earth's motion over the year to get a better picture of the actual orbit
        earth_positions = np.zeros((n_points, 3))
        
        print("Getting Earth's trajectory directly from SPICE for comparison...")
        for i, t in enumerate(np.linspace(start_et, start_et + duration, n_points)):
            earth_state_at_t, _ = spice.spkezr(str(earth_id), t, "ECLIPJ2000", "NONE", str(sun_id))
            earth_positions[i] = earth_state_at_t[:3]
        
        # Calculate orbital characteristics from the actual complete trajectory
        min_dist = np.min([np.linalg.norm(pos) for pos in earth_positions])
        max_dist = np.max([np.linalg.norm(pos) for pos in earth_positions])
        actual_sma = (min_dist + max_dist) / 2
        actual_ecc = (max_dist - min_dist) / (max_dist + min_dist)
        
        print("\nOrbital characteristics from SPICE trajectory:")
        print(f"Minimum distance: {min_dist:.2f} km")
        print(f"Maximum distance: {max_dist:.2f} km")
        print(f"Estimated semi-major axis: {actual_sma:.2f} km")
        print(f"Estimated eccentricity: {actual_ecc:.6f}")
        
        # Get the orbital plane normal vector
        orbital_plane_z = np.cross(earth_positions[0], earth_positions[25])
        orbital_plane_z = orbital_plane_z / np.linalg.norm(orbital_plane_z)
        
        # Plotting
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot Earth's simulated orbit
        ax.plot(earth_traj[:, 0], earth_traj[:, 1], earth_traj[:, 2], 'b-', label='Simulated orbit')
        
        # Plot Earth's SPICE trajectory
        ax.plot(earth_positions[:, 0], earth_positions[:, 1], earth_positions[:, 2], 'g--', label='SPICE trajectory')
        
        # Mark starting position
        ax.scatter(r0[0], r0[1], r0[2], c='blue', s=100, marker='o', label='Earth at start')
        
        # Add Sun at the center
        ax.scatter(0, 0, 0, c='yellow', s=200, marker='o', label='Sun')
        
        # Better axis scaling
        all_points = np.vstack([earth_traj, earth_positions, np.array([0, 0, 0])])  # Include Sun position
        
        max_val = np.max(np.abs(all_points))
        ax.set_xlim(-max_val*1.1, max_val*1.1)
        ax.set_ylim(-max_val*1.1, max_val*1.1)
        ax.set_zlim(-max_val*1.1, max_val*1.1)
        
        # Set labels and title
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title('Earth Orbit Around the Sun')
        
        # Add legend
        ax.legend()
        
        # Add orbital info to plot
        info_text = (
            f"Earth Orbital Elements:\n"
            f"Semi-major axis: {actual_sma:.2e} km\n"
            f"Eccentricity: {actual_ecc:.6f}\n"
            f"Period: {period_days:.2f} days"
        )
        plt.figtext(0.15, 0.15, info_text, fontsize=12, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Show the plot
        plt.tight_layout()
        plt.savefig('earth_orbit.png', dpi=300)
        print("Plot saved to earth_orbit.png")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("Unloading SPICE kernels...")
        spice.unload(lsk_path)
        spice.unload(spk_path)
        spice.unload(pck_path)

if __name__ == "__main__":
    simple_earth_orbit() 