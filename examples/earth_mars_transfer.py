#!/usr/bin/env python3
"""
Earth-to-Mars Transfer Trajectory

Simple and stable calculation of a Hohmann transfer from Earth to Mars.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
G = 6.67430e-11  # Gravitational constant, m^3/kg/s^2
M_SUN = 1.989e30  # Solar mass, kg
AU = 149.6e9  # Astronomical unit, m
DAY = 86400  # Seconds in a day

# Convert gravitational parameter to m^3/s^2
MU_SUN = G * M_SUN

# Orbital parameters (using SI units: meters and seconds)
EARTH_ORBITAL_RADIUS = 1.0 * AU  # m
MARS_ORBITAL_RADIUS = 1.524 * AU  # m

def hohmann_transfer_delta_v(r1, r2, mu):
    """
    Calculate the delta-v requirements for a Hohmann transfer.
    All inputs and outputs in SI units (meters and seconds).
    
    Args:
        r1: Initial orbital radius (m)
        r2: Final orbital radius (m)
        mu: Gravitational parameter (m^3/s^2)
    
    Returns:
        Dictionary with delta-v values and transfer parameters
    """
    # Calculate velocities in circular orbits
    v1 = np.sqrt(mu / r1)  # m/s
    v2 = np.sqrt(mu / r2)  # m/s
    
    # Semi-major axis of transfer orbit
    a_transfer = (r1 + r2) / 2  # m
    
    # Velocities in transfer orbit at perihelion and aphelion
    vt1 = np.sqrt(mu * (2/r1 - 1/a_transfer))  # m/s
    vt2 = np.sqrt(mu * (2/r2 - 1/a_transfer))  # m/s
    
    # Delta-v at departure and arrival
    delta_v1 = abs(vt1 - v1)  # m/s
    delta_v2 = abs(v2 - vt2)  # m/s
    total_delta_v = delta_v1 + delta_v2  # m/s
    
    # Time of flight (half the orbital period)
    tof = np.pi * np.sqrt(a_transfer**3 / mu)  # seconds
    
    return {
        'delta_v1': delta_v1,
        'delta_v2': delta_v2,
        'total_delta_v': total_delta_v,
        'tof': tof,
        'a_transfer': a_transfer,
        'v1': v1,
        'v2': v2,
        'vt1': vt1,
        'vt2': vt2
    }

def plot_hohmann_transfer(r1, r2, title, filename):
    """
    Generate a plot of the Hohmann transfer orbit.
    
    Args:
        r1: Initial orbital radius (m)
        r2: Final orbital radius (m)
        title: Plot title
        filename: Output file name
    """
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    
    # Plot Sun
    ax.scatter(0, 0, color='yellow', s=200, label='Sun', zorder=3)
    
    # Plot initial orbit (Earth)
    theta = np.linspace(0, 2*np.pi, 100)
    x1 = r1 * np.cos(theta)
    y1 = r1 * np.sin(theta)
    ax.plot(x1, y1, 'b-', label='Earth Orbit', zorder=1)
    
    # Plot final orbit (Mars)
    x2 = r2 * np.cos(theta)
    y2 = r2 * np.sin(theta)
    ax.plot(x2, y2, 'r-', label='Mars Orbit', zorder=1)
    
    # Define transfer orbit
    a_transfer = (r1 + r2) / 2
    c = a_transfer - r1  # Distance from center to focus
    b_transfer = np.sqrt(a_transfer**2 - c**2)  # Semi-minor axis
    
    # Plot transfer orbit
    theta_transfer = np.linspace(0, np.pi, 100)
    x_transfer = a_transfer * np.cos(theta_transfer) - c
    y_transfer = b_transfer * np.sin(theta_transfer)
    ax.plot(x_transfer, y_transfer, 'g--', label='Transfer Orbit', zorder=2)
    
    # Plot planets at departure and arrival
    ax.scatter(r1, 0, color='blue', s=100, label='Earth at Departure', zorder=3)
    ax.scatter(-r2, 0, color='red', s=100, label='Mars at Arrival', zorder=3)
    
    # Equal aspect ratio
    ax.set_aspect('equal')
    
    # Limits with some padding
    max_r = max(r1, r2) * 1.1
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    
    # Labels and title
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Distance (m)')
    ax.set_title(title)
    
    # Add legend
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Plot saved to {filename}")
    return fig

def calculate_launch_parameters():
    """
    Calculate and display Earth-to-Mars transfer parameters.
    """
    # Calculate Hohmann transfer parameters
    params = hohmann_transfer_delta_v(EARTH_ORBITAL_RADIUS, MARS_ORBITAL_RADIUS, MU_SUN)
    
    # Calculate Earth-specific parameters
    # Earth constants
    M_EARTH = 5.972e24  # Earth mass, kg
    R_EARTH = 6.371e6   # Earth radius, m
    MU_EARTH = G * M_EARTH
    
    # Low Earth Orbit (LEO) altitude
    leo_altitude = 300e3  # m
    leo_radius = R_EARTH + leo_altitude
    leo_velocity = np.sqrt(MU_EARTH / leo_radius)
    
    # Calculate hyperbolic excess velocity (v_inf)
    v_inf = params['delta_v1']
    
    # Calculate departure C3 (characteristic energy)
    c3 = v_inf**2
    
    # Calculate departure delta-v from LEO
    v_departure = np.sqrt(v_inf**2 + 2*MU_EARTH/leo_radius)
    delta_v_leo_to_escape = v_departure - leo_velocity
    
    # Mars-specific parameters
    M_MARS = 6.39e23  # Mars mass, kg
    R_MARS = 3.389e6  # Mars radius, m
    MU_MARS = G * M_MARS
    
    # Mars orbit
    mars_orbit_altitude = 400e3  # m
    mars_orbit_radius = R_MARS + mars_orbit_altitude
    mars_orbit_velocity = np.sqrt(MU_MARS / mars_orbit_radius)
    
    # Mars capture delta-v
    v_inf_mars = params['delta_v2']
    v_approach = np.sqrt(v_inf_mars**2 + 2*MU_MARS/mars_orbit_radius)
    delta_v_capture = v_approach - mars_orbit_velocity
    
    # Total mission delta-v
    total_mission_delta_v = delta_v_leo_to_escape + delta_v_capture
    
    # Print parameters
    print("Earth-to-Mars Hohmann Transfer Parameters")
    print("=========================================")
    print(f"Earth orbital radius: {EARTH_ORBITAL_RADIUS/AU:.3f} AU")
    print(f"Mars orbital radius: {MARS_ORBITAL_RADIUS/AU:.3f} AU")
    print(f"Transfer orbit semi-major axis: {params['a_transfer']/AU:.3f} AU")
    print(f"Time of flight: {params['tof']/DAY:.1f} days")
    
    print("\nHeliocentric Velocities:")
    print(f"Earth orbital velocity: {params['v1']/1000:.2f} km/s")
    print(f"Spacecraft velocity at Earth departure: {params['vt1']/1000:.2f} km/s")
    print(f"Spacecraft velocity at Mars arrival: {params['vt2']/1000:.2f} km/s")
    print(f"Mars orbital velocity: {params['v2']/1000:.2f} km/s")
    
    print("\nDelta-V Requirements:")
    print(f"Heliocentric departure delta-v: {params['delta_v1']/1000:.2f} km/s")
    print(f"Heliocentric arrival delta-v: {params['delta_v2']/1000:.2f} km/s")
    print(f"Heliocentric total delta-v: {params['total_delta_v']/1000:.2f} km/s")
    
    print("\nEarth Departure:")
    print(f"LEO altitude: {leo_altitude/1000:.1f} km")
    print(f"LEO velocity: {leo_velocity/1000:.2f} km/s")
    print(f"Hyperbolic excess velocity (v_inf): {v_inf/1000:.2f} km/s")
    print(f"C3 (characteristic energy): {c3/1e6:.2f} km²/s²")
    print(f"Trans-Mars Injection delta-v (from LEO): {delta_v_leo_to_escape/1000:.2f} km/s")
    
    print("\nMars Arrival:")
    print(f"Mars orbit altitude: {mars_orbit_altitude/1000:.1f} km")
    print(f"Mars orbit velocity: {mars_orbit_velocity/1000:.2f} km/s")
    print(f"Hyperbolic approach velocity: {v_inf_mars/1000:.2f} km/s")
    print(f"Mars Orbit Insertion delta-v: {delta_v_capture/1000:.2f} km/s")
    
    print("\nTotal Mission Delta-V:")
    print(f"Earth departure (TMI from LEO): {delta_v_leo_to_escape/1000:.2f} km/s")
    print(f"Mars arrival (MOI): {delta_v_capture/1000:.2f} km/s")
    print(f"Total required: {total_mission_delta_v/1000:.2f} km/s")
    
    # Return all calculated parameters for plot title
    return params, delta_v_leo_to_escape, delta_v_capture

def plot_3d_transfer():
    """
    Create a 3D plot of the Earth-to-Mars transfer trajectory.
    """
    # Figure setup
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate orbits
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Earth orbit
    x_earth = EARTH_ORBITAL_RADIUS * np.cos(theta)
    y_earth = EARTH_ORBITAL_RADIUS * np.sin(theta)
    z_earth = np.zeros_like(theta)
    
    # Mars orbit
    x_mars = MARS_ORBITAL_RADIUS * np.cos(theta)
    y_mars = MARS_ORBITAL_RADIUS * np.sin(theta)
    z_mars = np.zeros_like(theta)
    
    # Transfer orbit parameters
    a_transfer = (EARTH_ORBITAL_RADIUS + MARS_ORBITAL_RADIUS) / 2
    c = a_transfer - EARTH_ORBITAL_RADIUS
    b_transfer = np.sqrt(a_transfer**2 - c**2)
    
    # Transfer orbit
    theta_transfer = np.linspace(0, np.pi, 100)
    x_transfer = a_transfer * np.cos(theta_transfer) - c
    y_transfer = b_transfer * np.sin(theta_transfer)
    z_transfer = np.zeros_like(theta_transfer)
    
    # Plot orbits
    ax.plot(x_earth, y_earth, z_earth, 'b-', label="Earth's Orbit", alpha=0.7)
    ax.plot(x_mars, y_mars, z_mars, 'r-', label="Mars's Orbit", alpha=0.7)
    ax.plot(x_transfer, y_transfer, z_transfer, 'g--', linewidth=2, label='Transfer Trajectory')
    
    # Plot celestial bodies
    ax.scatter(0, 0, 0, color='yellow', s=200, label='Sun')
    ax.scatter(EARTH_ORBITAL_RADIUS, 0, 0, color='blue', s=100, label='Earth at Departure')
    ax.scatter(-MARS_ORBITAL_RADIUS, 0, 0, color='red', s=100, label='Mars at Arrival')
    
    # Set axis labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Set equal aspect ratio
    # Need to set limits with some padding first
    max_val = MARS_ORBITAL_RADIUS * 1.1
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_zlim(-max_val, max_val)
    
    # Add a grid
    ax.grid(True)
    
    # Add title and legend
    ax.set_title('Earth-to-Mars Hohmann Transfer Trajectory')
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig("earth_mars_3d.png", dpi=300)
    print("3D plot saved to earth_mars_3d.png")
    
    return fig

def main():
    """Main function"""
    try:
        # Calculate mission parameters
        params, delta_v_earth, delta_v_mars = calculate_launch_parameters()
        
        # Generate Hohmann transfer plot
        title = (f"Earth-Mars Hohmann Transfer Orbit\n"
                 f"Time of Flight: {params['tof']/DAY:.1f} days, "
                 f"Total Mission ΔV: {(delta_v_earth + delta_v_mars)/1000:.2f} km/s")
        plot_hohmann_transfer(EARTH_ORBITAL_RADIUS, MARS_ORBITAL_RADIUS, title, "earth_mars_transfer.png")
        
        # Generate 3D plot
        plot_3d_transfer()
        
        print("\nSimulation completed successfully!")
        
    except Exception as e:
        print(f"Error in simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 