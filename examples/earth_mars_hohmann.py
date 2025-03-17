#!/usr/bin/env python3
"""
Earth-to-Mars Hohmann Transfer Simulator

This script calculates and visualizes a Hohmann transfer orbit from Earth to Mars.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime

# Constants
AU = 149.6e6  # Astronomical Unit in km
SUN_MU = 1.32712440018e20  # Sun's gravitational parameter in km^3/s^2
EARTH_MU = 3.986004418e14  # Earth's gravitational parameter in km^3/s^2
MARS_MU = 4.282837e13  # Mars's gravitational parameter in km^3/s^2
DAY = 86400  # Seconds in a day

# Define orbital parameters of planets (simplified circular orbits)
EARTH_ORBIT = {
    'a': 1.0 * AU,         # Semi-major axis (km)
    'e': 0.0,              # Eccentricity (circular approximation)
    'i': 0.0,              # Inclination (rad)
    'period': 365.25 * DAY # Period (seconds)
}

MARS_ORBIT = {
    'a': 1.524 * AU,       # Semi-major axis (km)
    'e': 0.0,              # Eccentricity (circular approximation)
    'i': 0.0,              # Inclination (rad)
    'period': 686.98 * DAY # Period (seconds)
}

def get_planet_position(orbit, angle):
    """
    Calculate position of a planet in a circular orbit at a given angle
    
    Args:
        orbit: Dictionary with orbital parameters
        angle: Orbital angle in radians
    
    Returns:
        Position vector [x, y, z] in km
    """
    radius = orbit['a']
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = 0
    return np.array([x, y, z])

def get_planet_velocity(orbit, angle):
    """
    Calculate velocity of a planet in a circular orbit at a given angle
    
    Args:
        orbit: Dictionary with orbital parameters
        angle: Orbital angle in radians
    
    Returns:
        Velocity vector [vx, vy, vz] in km/s
    """
    radius = orbit['a']
    speed = np.sqrt(SUN_MU / radius)
    
    # Velocity is perpendicular to position vector in circular orbit
    vx = -speed * np.sin(angle)
    vy = speed * np.cos(angle)
    vz = 0
    return np.array([vx, vy, vz])

def calculate_hohmann_transfer(r1, r2):
    """
    Calculate parameters for a Hohmann transfer between circular orbits
    
    Args:
        r1: Radius of departure orbit (km)
        r2: Radius of arrival orbit (km)
    
    Returns:
        Dictionary with transfer parameters
    """
    # Semi-major axis of transfer orbit
    a_t = (r1 + r2) / 2
    
    # Period of transfer orbit (half orbit)
    T_t = np.pi * np.sqrt(a_t**3 / SUN_MU)
    
    # Velocities in circular orbits
    v1 = np.sqrt(SUN_MU / r1)
    v2 = np.sqrt(SUN_MU / r2)
    
    # Velocities in transfer orbit at perihelion and aphelion
    vt_p = np.sqrt(SUN_MU * (2/r1 - 1/a_t))  # at perihelion (Earth departure)
    vt_a = np.sqrt(SUN_MU * (2/r2 - 1/a_t))  # at aphelion (Mars arrival)
    
    # Delta-V calculations
    dv1 = abs(vt_p - v1)  # Earth departure
    dv2 = abs(v2 - vt_a)  # Mars arrival
    
    return {
        'a_transfer': a_t,
        'tof': T_t,
        'delta_v1': dv1,
        'delta_v2': dv2,
        'total_delta_v': dv1 + dv2,
        'v1': v1,
        'v2': v2,
        'vt_p': vt_p,
        'vt_a': vt_a
    }

def propagate_hohmann(r1, v1, r2, time_points):
    """
    Propagate a state vector along a Hohmann transfer orbit
    
    Args:
        r1: Initial position vector [x, y, z] (km)
        v1: Initial velocity vector [vx, vy, vz] (km/s)
        r2: Final position vector [x, y, z] (km)
        time_points: Array of time points (seconds from start)
    
    Returns:
        Array of position vectors at each time point
    """
    # Calculate orbital elements from initial state
    r1_mag = np.linalg.norm(r1)
    v1_mag = np.linalg.norm(v1)
    
    # Calculate specific energy
    energy = v1_mag**2 / 2 - SUN_MU / r1_mag
    
    # Calculate semi-major axis
    a = -SUN_MU / (2 * energy)
    
    # Calculate angular momentum vector
    h_vec = np.cross(r1, v1)
    h_mag = np.linalg.norm(h_vec)
    
    # Calculate eccentricity vector
    e_vec = np.cross(v1, h_vec) / SUN_MU - r1 / r1_mag
    e = np.linalg.norm(e_vec)
    
    # Calculate true anomaly at departure point
    cos_nu = np.dot(e_vec, r1) / (e * r1_mag)
    # Handle numerical issues
    if cos_nu > 1:
        cos_nu = 1
    elif cos_nu < -1:
        cos_nu = -1
    nu = np.arccos(cos_nu)
    if np.dot(r1, v1) < 0:
        nu = 2 * np.pi - nu
    
    # Calculate period
    period = 2 * np.pi * np.sqrt(a**3 / SUN_MU)
    
    # Calculate eccentric anomaly at departure
    E = 2 * np.arctan(np.sqrt((1-e)/(1+e)) * np.tan(nu/2))
    
    # Calculate mean anomaly at departure
    M0 = E - e * np.sin(E)
    
    # Calculate mean motion
    n = 2 * np.pi / period
    
    # Initialize trajectory array
    trajectory = np.zeros((len(time_points), 3))
    
    # Propagate orbit
    for i, t in enumerate(time_points):
        # Calculate mean anomaly at time t
        M = M0 + n * t
        
        # Solve Kepler's equation for eccentric anomaly
        E = M
        for _ in range(100):
            E_next = M + e * np.sin(E)
            if abs(E_next - E) < 1e-8:
                E = E_next
                break
            E = E_next
            
        # Calculate true anomaly
        cos_nu = (np.cos(E) - e) / (1 - e * np.cos(E))
        sin_nu = (np.sqrt(1 - e**2) * np.sin(E)) / (1 - e * np.cos(E))
        nu = np.arctan2(sin_nu, cos_nu)
        
        # Calculate position in orbit
        r = a * (1 - e * np.cos(E))
        x = r * np.cos(nu)
        y = r * np.sin(nu)
        z = 0
        
        trajectory[i] = np.array([x, y, z])
    
    return trajectory

def calculate_earth_departure_maneuver(v1_earth, v1_sc):
    """
    Calculate Earth departure maneuver parameters
    
    Args:
        v1_earth: Earth's heliocentric velocity (km/s)
        v1_sc: Spacecraft's heliocentric velocity after departure (km/s)
    
    Returns:
        Dictionary with maneuver parameters
    """
    # Velocity difference in heliocentric frame
    delta_v_helio = np.linalg.norm(v1_sc - v1_earth)
    
    # Earth parameters
    earth_radius = 6378.0  # km
    parking_altitude = 300.0  # km
    parking_radius = earth_radius + parking_altitude
    
    # Circular parking orbit velocity
    v_parking = np.sqrt(EARTH_MU / parking_radius)
    
    # Calculate v-infinity (hyperbolic excess velocity)
    v_inf = delta_v_helio
    
    # Calculate C3 (characteristic energy)
    c3 = v_inf**2
    
    # Calculate departure burn delta-v from parking orbit
    v_departure = np.sqrt(v_inf**2 + 2 * EARTH_MU / parking_radius)
    delta_v_departure = v_departure - v_parking
    
    return {
        'delta_v_helio': delta_v_helio,
        'v_parking': v_parking,
        'v_inf': v_inf,
        'c3': c3,
        'delta_v_departure': delta_v_departure
    }

def calculate_mars_arrival_maneuver(v2_mars, v2_sc):
    """
    Calculate Mars arrival maneuver parameters
    
    Args:
        v2_mars: Mars's heliocentric velocity (km/s)
        v2_sc: Spacecraft's heliocentric velocity at arrival (km/s)
    
    Returns:
        Dictionary with maneuver parameters
    """
    # Velocity difference in heliocentric frame
    delta_v_helio = np.linalg.norm(v2_sc - v2_mars)
    
    # Mars parameters
    mars_radius = 3396.0  # km
    mars_orbit_altitude = 400.0  # km
    mars_orbit_radius = mars_radius + mars_orbit_altitude
    
    # Circular Mars orbit velocity
    v_mars_orbit = np.sqrt(MARS_MU / mars_orbit_radius)
    
    # Calculate v-infinity (hyperbolic approach velocity)
    v_inf = delta_v_helio
    
    # Calculate Mars orbit insertion delta-v
    v_approach = np.sqrt(v_inf**2 + 2 * MARS_MU / mars_orbit_radius)
    delta_v_insertion = v_approach - v_mars_orbit
    
    return {
        'delta_v_helio': delta_v_helio,
        'v_mars_orbit': v_mars_orbit,
        'v_inf': v_inf,
        'delta_v_insertion': delta_v_insertion
    }

def plot_transfer(earth_pos, mars_pos, trajectory, title, output_file):
    """
    Plot the Hohmann transfer trajectory
    
    Args:
        earth_pos: Earth's position at departure
        mars_pos: Mars's position at arrival
        trajectory: Transfer orbit trajectory
        title: Plot title
        output_file: Output file name
    """
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot orbits
    theta = np.linspace(0, 2*np.pi, 100)
    earth_orbit_x = EARTH_ORBIT['a'] * np.cos(theta)
    earth_orbit_y = EARTH_ORBIT['a'] * np.sin(theta)
    earth_orbit_z = np.zeros_like(theta)
    
    mars_orbit_x = MARS_ORBIT['a'] * np.cos(theta)
    mars_orbit_y = MARS_ORBIT['a'] * np.sin(theta)
    mars_orbit_z = np.zeros_like(theta)
    
    ax.plot(earth_orbit_x, earth_orbit_y, earth_orbit_z, 'b--', label="Earth's Orbit", alpha=0.7)
    ax.plot(mars_orbit_x, mars_orbit_y, mars_orbit_z, 'r--', label="Mars's Orbit", alpha=0.7)
    
    # Plot transfer trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 'g-', label='Transfer Trajectory', linewidth=2)
    
    # Plot Sun, Earth, and Mars
    ax.scatter(0, 0, 0, color='yellow', s=200, label='Sun')
    ax.scatter(earth_pos[0], earth_pos[1], earth_pos[2], color='blue', s=100, label='Earth at Departure')
    ax.scatter(mars_pos[0], mars_pos[1], mars_pos[2], color='red', s=100, label='Mars at Arrival')
    
    # Mark departure and arrival points
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], color='green', s=80, marker='^', label='Departure')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], color='green', s=80, marker='v', label='Arrival')
    
    # Set equal aspect ratio
    max_val = max(np.max(MARS_ORBIT['a']), np.max(np.abs(trajectory)))
    ax.set_xlim(-max_val*1.2, max_val*1.2)
    ax.set_ylim(-max_val*1.2, max_val*1.2)
    ax.set_zlim(-max_val*1.2, max_val*1.2)
    
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    
    return fig

def main():
    """Main function to run the simulation"""
    print("Earth-to-Mars Hohmann Transfer Simulator")
    print("=======================================")
    
    try:
        # Calculate angles for optimal Hohmann transfer
        # For a Hohmann transfer, the departure angle and arrival angle
        # have a specific relationship for circular, coplanar orbits
        earth_departure_angle = 0  # Starting position
        
        # Calculate phase angle for Hohmann transfer
        # Mars should be ahead of Earth by this angle at departure
        phase_angle = np.pi * (1 - np.sqrt((EARTH_ORBIT['a'] / MARS_ORBIT['a'])**3))
        
        # Mars will be at this angle when spacecraft arrives
        mars_arrival_angle = earth_departure_angle + phase_angle
        
        # Calculate positions
        earth_pos = get_planet_position(EARTH_ORBIT, earth_departure_angle)
        mars_pos = get_planet_position(MARS_ORBIT, mars_arrival_angle)
        
        # Calculate velocities
        earth_vel = get_planet_velocity(EARTH_ORBIT, earth_departure_angle)
        mars_vel = get_planet_velocity(MARS_ORBIT, mars_arrival_angle)
        
        # Calculate Hohmann transfer parameters
        hohmann = calculate_hohmann_transfer(EARTH_ORBIT['a'], MARS_ORBIT['a'])
        transfer_time = hohmann['tof']
        
        # Departure and arrival velocity vectors
        v_sc_dep = np.array([
            -hohmann['vt_p'] * np.sin(earth_departure_angle),
            hohmann['vt_p'] * np.cos(earth_departure_angle),
            0
        ])
        
        v_sc_arr = np.array([
            -hohmann['vt_a'] * np.sin(mars_arrival_angle),
            hohmann['vt_a'] * np.cos(mars_arrival_angle),
            0
        ])
        
        # Calculate detailed maneuvers
        departure = calculate_earth_departure_maneuver(earth_vel, v_sc_dep)
        arrival = calculate_mars_arrival_maneuver(mars_vel, v_sc_arr)
        
        # Time points for propagation
        time_points = np.linspace(0, transfer_time, 100)
        
        # Propagate the transfer orbit
        trajectory = propagate_hohmann(earth_pos, v_sc_dep, mars_pos, time_points)
        
        # Print mission details
        print("\n===== MISSION PARAMETERS =====")
        print(f"Earth departure radius: {EARTH_ORBIT['a']/AU:.3f} AU")
        print(f"Mars arrival radius: {MARS_ORBIT['a']/AU:.3f} AU")
        print(f"Transfer orbit semi-major axis: {hohmann['a_transfer']/AU:.3f} AU")
        print(f"Transfer time: {transfer_time/DAY:.1f} days")
        
        print("\nVelocities:")
        print(f"Earth orbital velocity: {hohmann['v1']:.2f} km/s")
        print(f"Spacecraft velocity at departure: {hohmann['vt_p']:.2f} km/s")
        print(f"Spacecraft velocity at arrival: {hohmann['vt_a']:.2f} km/s")
        print(f"Mars orbital velocity: {hohmann['v2']:.2f} km/s")
        
        print("\nDelta-V requirements:")
        print(f"Heliocentric departure ΔV: {departure['delta_v_helio']:.2f} km/s")
        print(f"Earth departure burn (from 300 km parking orbit): {departure['delta_v_departure']:.2f} km/s")
        print(f"C3 (characteristic energy): {departure['c3']:.2f} km²/s²")
        
        print(f"Heliocentric arrival ΔV: {arrival['delta_v_helio']:.2f} km/s")
        print(f"Mars orbit insertion burn (into 400 km orbit): {arrival['delta_v_insertion']:.2f} km/s")
        
        print(f"Total mission ΔV: {departure['delta_v_departure'] + arrival['delta_v_insertion']:.2f} km/s")
        
        # Plot the transfer
        title = f"Earth-Mars Hohmann Transfer Orbit\nTOF: {transfer_time/DAY:.1f} days, " \
                f"Total ΔV: {departure['delta_v_departure'] + arrival['delta_v_insertion']:.2f} km/s"
        plot_transfer(earth_pos, mars_pos, trajectory, title, "earth_mars_hohmann.png")
        
        print("\nSimulation completed successfully!")
        
    except Exception as e:
        print(f"Error in simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 