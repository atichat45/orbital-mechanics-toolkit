#!/usr/bin/env python3
"""
Interplanetary Hohmann Transfer Analysis

This script analyzes and compares Hohmann transfer trajectories from Earth to all other
planets in the solar system, including trajectory length, travel time, and required delta-V.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
from scipy.integrate import solve_ivp
import math
from matplotlib.ticker import ScalarFormatter

# Set better plotting defaults for readability
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

# Constants
G = 6.67430e-11  # Gravitational constant, m^3/kg/s^2
M_SUN = 1.989e30  # Solar mass, kg
AU = 149.6e9  # Astronomical unit, m
DAY = 86400  # Seconds in a day
YEAR = 365.25 * DAY  # Seconds in a year

# Convert gravitational parameter to m^3/s^2
MU_SUN = G * M_SUN

# Planetary data (average orbital elements and physical properties)
PLANETS = {
    'Mercury': {
        'a': 0.387 * AU,  # Semi-major axis (m)
        'e': 0.206,       # Eccentricity
        'i': np.radians(7.0),  # Inclination (rad)
        'period': 88.0 * DAY,  # Orbital period (s)
        'radius': 2.44e6,  # Planet radius (m)
        'mass': 3.3011e23,  # Planet mass (kg)
        'color': 'gray',
        'z_offset': 0.04 * AU,  # For visualization
    },
    'Venus': {
        'a': 0.723 * AU,
        'e': 0.007,
        'i': np.radians(3.4),
        'period': 225.0 * DAY,
        'radius': 6.052e6,
        'mass': 4.8675e24,
        'color': 'gold',
        'z_offset': 0.02 * AU,
    },
    'Earth': {
        'a': 1.0 * AU,
        'e': 0.017,
        'i': np.radians(0.0),
        'period': 365.25 * DAY,
        'radius': 6.371e6,
        'mass': 5.972e24,
        'color': 'blue',
        'z_offset': 0.0,
    },
    'Mars': {
        'a': 1.524 * AU,
        'e': 0.093,
        'i': np.radians(1.9),
        'period': 687.0 * DAY,
        'radius': 3.39e6,
        'mass': 6.417e23,
        'color': 'red',
        'z_offset': 0.03 * AU,
    },
    'Jupiter': {
        'a': 5.203 * AU,
        'e': 0.048,
        'i': np.radians(1.3),
        'period': 11.86 * YEAR,
        'radius': 6.9911e7,
        'mass': 1.898e27,
        'color': 'orange',
        'z_offset': 0.05 * AU,
    },
    'Saturn': {
        'a': 9.537 * AU,
        'e': 0.054,
        'i': np.radians(2.5),
        'period': 29.46 * YEAR,
        'radius': 5.8232e7,
        'mass': 5.683e26,
        'color': 'khaki',
        'z_offset': 0.08 * AU,
    },
    'Uranus': {
        'a': 19.191 * AU,
        'e': 0.047,
        'i': np.radians(0.8),
        'period': 84.01 * YEAR,
        'radius': 2.5362e7,
        'mass': 8.681e25,
        'color': 'skyblue',
        'z_offset': 0.1 * AU,
    },
    'Neptune': {
        'a': 30.069 * AU,
        'e': 0.009,
        'i': np.radians(1.8),
        'period': 164.8 * YEAR,
        'radius': 2.4622e7,
        'mass': 1.024e26,
        'color': 'blue',
        'z_offset': 0.15 * AU,
    }
}

def hohmann_transfer_analytical(r1, r2):
    """
    Calculate analytical properties of Hohmann transfer between circular orbits.
    
    Args:
        r1: Radius of departure orbit (m)
        r2: Radius of arrival orbit (m)
    
    Returns:
        Dictionary with transfer parameters including delta-v, time of flight, and path length
    """
    # Semi-major axis of transfer orbit
    a_transfer = (r1 + r2) / 2
    
    # Velocities in circular orbits
    v1 = np.sqrt(MU_SUN / r1)
    v2 = np.sqrt(MU_SUN / r2)
    
    # Velocities in transfer orbit at periapsis and apoapsis
    vt1 = np.sqrt(MU_SUN * (2/r1 - 1/a_transfer))
    vt2 = np.sqrt(MU_SUN * (2/r2 - 1/a_transfer))
    
    # Delta-v requirements
    dv1 = abs(vt1 - v1)
    dv2 = abs(v2 - vt2)
    total_dv = dv1 + dv2
    
    # Time of flight (half the orbital period)
    tof = np.pi * np.sqrt(a_transfer**3 / MU_SUN)
    
    # Calculate path length (half-ellipse circumference approximation)
    # Using Ramanujan's approximation for ellipse circumference
    a = a_transfer
    b = np.sqrt(r1 * r2)  # Semi-minor axis for transfer orbit
    h = ((a - b) / (a + b))**2
    path_length = np.pi * (a + b) * (1 + (3*h)/(10 + np.sqrt(4 - 3*h)))
    half_path_length = path_length / 2  # For half the ellipse
    
    return {
        'a_transfer': a_transfer,
        'tof': tof,
        'total_dv': total_dv,
        'dv1': dv1,
        'dv2': dv2,
        'v1': v1,
        'v2': v2,
        'vt1': vt1,
        'vt2': vt2,
        'path_length': half_path_length
    }

def two_body_equation(t, y, mu):
    """
    Two-body equation of motion for numerical integration.
    
    Args:
        t: Time
        y: State vector [x, y, z, vx, vy, vz]
        mu: Gravitational parameter
    
    Returns:
        Derivatives of the state vector
    """
    r = y[0:3]
    v = y[3:6]
    
    # Calculate norm of radius vector
    r_norm = np.linalg.norm(r)
    
    # Acceleration due to gravity
    a = -mu * r / r_norm**3
    
    return np.concatenate([v, a])

def propagate_orbit_numerical(r0, v0, tof, mu=MU_SUN, steps=1000):
    """
    Numerically propagate orbit using two-body dynamics.
    
    Args:
        r0: Initial position vector (m)
        v0: Initial velocity vector (m/s)
        tof: Time of flight (s)
        mu: Gravitational parameter (m^3/s^2)
        steps: Number of time steps for propagation
        
    Returns:
        Dictionary with trajectory points and path length
    """
    # Initial state vector
    y0 = np.concatenate([r0, v0])
    
    # Time points
    t_span = (0, tof)
    t_eval = np.linspace(0, tof, steps)
    
    # Integrate equations of motion
    sol = solve_ivp(
        lambda t, y: two_body_equation(t, y, mu),
        t_span,
        y0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-8
    )
    
    # Extract trajectory points
    positions = sol.y[0:3].T
    velocities = sol.y[3:6].T
    
    # Calculate path length
    path_length = 0
    for i in range(1, len(positions)):
        segment_length = np.linalg.norm(positions[i] - positions[i-1])
        path_length += segment_length
    
    return {
        'times': sol.t,
        'positions': positions,
        'velocities': velocities,
        'path_length': path_length
    }

def get_departure_state(planet_name, departure_angle=0):
    """
    Get the position and velocity of a planet at a given orbital angle.
    
    Args:
        planet_name: Name of the planet
        departure_angle: Angle in radians
    
    Returns:
        Tuple of position and velocity vectors
    """
    planet = PLANETS[planet_name]
    a = planet['a']
    e = planet['e']
    i = planet['i']
    
    # For simplification, assume circular coplanar orbit
    r = a * (1 - e**2) / (1 + e * np.cos(departure_angle))
    
    # Position in orbital plane
    x = r * np.cos(departure_angle)
    y = r * np.sin(departure_angle)
    z = r * np.sin(departure_angle) * np.sin(i)  # Simple inclination effect
    
    # For circular orbit approximation
    v_mag = np.sqrt(MU_SUN / r)
    
    # Velocity perpendicular to position vector in orbital plane
    vx = -v_mag * np.sin(departure_angle)
    vy = v_mag * np.cos(departure_angle)
    vz = 0  # Simplified
    
    return np.array([x, y, z]), np.array([vx, vy, vz])

def calculate_all_transfers():
    """
    Calculate transfer parameters for Earth to all other planets.
    
    Returns:
        DataFrame with comparison metrics for all transfers
    """
    results = []
    earth = PLANETS['Earth']
    earth_r, earth_v = get_departure_state('Earth')
    
    destinations = [p for p in PLANETS.keys() if p != 'Earth']
    
    for planet_name in destinations:
        planet = PLANETS[planet_name]
        
        # For simplicity, use semi-major axis as orbital radius
        # This assumes circular orbits as a first approximation
        r1 = earth['a'] 
        r2 = planet['a']
        
        # Calculate analytical solution
        analytical = hohmann_transfer_analytical(r1, r2)
        
        # Get target position at arrival
        # For a simplified model, assume optimal phase angle for Hohmann transfer
        if r2 > r1:  # Outbound
            planet_r, planet_v = get_departure_state(planet_name, departure_angle=np.pi)
        else:  # Inbound
            planet_r, planet_v = get_departure_state(planet_name, departure_angle=0)
        
        # Initial velocity vector for transfer orbit
        if r2 > r1:  # Outbound
            v_transfer = earth_v.copy()
            v_transfer[0] += analytical['dv1'] * np.sign(earth_v[0])
        else:  # Inbound
            v_transfer = earth_v.copy()
            v_transfer[0] -= analytical['dv1'] * np.sign(earth_v[0])
        
        # Numerical propagation
        numerical = propagate_orbit_numerical(earth_r, v_transfer, analytical['tof'])
        
        # Store results
        results.append({
            'target': planet_name,
            'a_target': r2,
            'a_transfer': analytical['a_transfer'],
            'tof_days': analytical['tof'] / DAY,
            'analytical_path_length': analytical['path_length'],
            'numerical_path_length': numerical['path_length'],
            'delta_v': analytical['total_dv'],
            'departure_dv': analytical['dv1'],
            'arrival_dv': analytical['dv2'],
            'numerical_trajectory': numerical['positions'],
            'numerical_times': numerical['times']
        })
    
    return pd.DataFrame(results)

def plot_transfer_trajectories(results_df):
    """
    Create a 3D plot of all transfer trajectories.
    
    Args:
        results_df: DataFrame with transfer results
    
    Returns:
        Figure object
    """
    # Create figure - INCREASED SIZE for better visibility
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Sun
    ax.scatter([0], [0], [0], color='yellow', s=500, label='Sun')
    
    # Plot Earth's orbit
    earth = PLANETS['Earth']
    theta = np.linspace(0, 2*np.pi, 100)
    x_earth = earth['a'] * np.cos(theta)
    y_earth = earth['a'] * np.sin(theta)
    z_earth = np.zeros_like(theta)
    ax.plot(x_earth, y_earth, z_earth, color=earth['color'], label="Earth's Orbit", alpha=0.7, linewidth=3)
    
    # Plot Earth
    ax.scatter([earth['a']], [0], [0], color=earth['color'], s=200, label='Earth')
    
    # Plot each planet's orbit with slight z offset for visibility
    for planet_name, planet in PLANETS.items():
        if planet_name == 'Earth':
            continue
            
        x_planet = planet['a'] * np.cos(theta)
        y_planet = planet['a'] * np.sin(theta)
        z_planet = np.zeros_like(theta) + planet['z_offset']  # Small offset for visibility
        
        ax.plot(x_planet, y_planet, z_planet, color=planet['color'], 
                label=f"{planet_name}'s Orbit", alpha=0.7, linestyle='--', linewidth=2)
        
        # Plot planet (at arbitrary position)
        planet_angle = np.pi  # Opposite side from Earth for visualization
        planet_x = planet['a'] * np.cos(planet_angle)
        planet_y = planet['a'] * np.sin(planet_angle)
        planet_z = planet['z_offset']
        ax.scatter([planet_x], [planet_y], [planet_z], color=planet['color'], s=200, label=planet_name)
    
    # Plot transfer trajectories
    for i, row in results_df.iterrows():
        trajectory = row['numerical_trajectory']
        planet_name = row['target']
        planet_color = PLANETS[planet_name]['color']
        
        # Add slight vertical offset for better visibility
        trajectory_with_offset = trajectory.copy()
        trajectory_with_offset[:, 2] += PLANETS[planet_name]['z_offset']
        
        ax.plot(trajectory_with_offset[:, 0], trajectory_with_offset[:, 1], trajectory_with_offset[:, 2], 
                label=f"Transfer to {planet_name}", linewidth=3)
        
        # Mark departure and arrival points
        ax.scatter(trajectory_with_offset[0, 0], trajectory_with_offset[0, 1], trajectory_with_offset[0, 2], 
                  color='green', s=100, marker='^')
        ax.scatter(trajectory_with_offset[-1, 0], trajectory_with_offset[-1, 1], trajectory_with_offset[-1, 2], 
                  color='red', s=100, marker='v')
    
    # Set axis limits based on the furthest planet
    max_radius = max(planet['a'] for planet in PLANETS.values()) * 1.1
    max_value = max_radius / AU
    
    # Use logarithmic scale for better visualization
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_zlim(-max_radius/4, max_radius/4)  # Smaller z range
    
    # Convert axis values to AU and add labels
    ticks = np.linspace(-max_radius, max_radius, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(np.linspace(-max_radius/4, max_radius/4, 5))
    
    # Format tick labels to show AU
    ax.set_xticklabels([f'{t/AU:.1f}' for t in ticks])
    ax.set_yticklabels([f'{t/AU:.1f}' for t in ticks])
    ax.set_zticklabels([f'{t/AU:.1f}' for t in np.linspace(-max_radius/4, max_radius/4, 5)])
    
    ax.set_xlabel('X (AU)', fontweight='bold')
    ax.set_ylabel('Y (AU)', fontweight='bold')
    ax.set_zlabel('Z (AU)', fontweight='bold')
    ax.set_title('Hohmann Transfer Trajectories from Earth to Other Planets', fontweight='bold', pad=20)
    
    # Improve legend - separate into two legends for better clarity
    # First legend for planets and Sun
    planet_handles = []
    planet_labels = []
    # Second legend for transfers
    transfer_handles = []
    transfer_labels = []
    
    handles, labels = ax.get_legend_handles_labels()
    for h, l in zip(handles, labels):
        if "Transfer" in l:
            transfer_handles.append(h)
            transfer_labels.append(l)
        elif "Orbit" not in l:
            planet_handles.append(h)
            planet_labels.append(l)
    
    # Place the legends in better positions
    ax.legend(planet_handles, planet_labels, loc='upper left', title="Celestial Bodies", frameon=True, framealpha=0.9)
    # Add second legend for transfers
    second_legend = ax.figure.legend(transfer_handles, transfer_labels, loc='upper right', title="Transfer Trajectories", 
                                      frameon=True, framealpha=0.9)
    ax.figure.add_artist(second_legend)
    
    # Add a grid for better reference
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Improve the view angle for better visualization
    ax.view_init(elev=25, azim=45)
    
    # Handle tight layout warnings safely
    try:
        plt.tight_layout()
    except:
        print("Warning: Tight layout could not be applied to 3D plot. Continuing anyway.")
    
    plt.savefig('hohmann_transfers_3d.png', dpi=300, bbox_inches='tight')
    print("3D plot saved as 'hohmann_transfers_3d.png'")
    
    return fig

def create_inner_planets_plot(results_df):
    """
    Create a zoomed-in plot of inner planet transfers.
    
    Args:
        results_df: DataFrame with transfer results
    
    Returns:
        Figure object
    """
    # Filter for inner planets
    inner_planets = ['Mercury', 'Venus', 'Mars']
    inner_df = results_df[results_df['target'].isin(inner_planets)]
    
    # Create figure - INCREASED SIZE
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Sun
    ax.scatter([0], [0], [0], color='yellow', s=300, label='Sun')
    
    # Plot Earth's orbit
    earth = PLANETS['Earth']
    theta = np.linspace(0, 2*np.pi, 100)
    x_earth = earth['a'] * np.cos(theta)
    y_earth = earth['a'] * np.sin(theta)
    z_earth = np.zeros_like(theta)
    ax.plot(x_earth, y_earth, z_earth, color=earth['color'], label="Earth's Orbit", alpha=0.7, linewidth=3)
    
    # Plot Earth
    ax.scatter([earth['a']], [0], [0], color=earth['color'], s=150, label='Earth')
    
    # Plot inner planets' orbits
    for planet_name in inner_planets:
        planet = PLANETS[planet_name]
        
        x_planet = planet['a'] * np.cos(theta)
        y_planet = planet['a'] * np.sin(theta)
        z_planet = np.zeros_like(theta) + planet['z_offset']
        
        ax.plot(x_planet, y_planet, z_planet, color=planet['color'], 
                label=f"{planet_name}'s Orbit", alpha=0.7, linestyle='--', linewidth=2)
        
        # Plot planet
        planet_angle = np.pi if planet['a'] > earth['a'] else 0  # Arbitrary positioning
        planet_x = planet['a'] * np.cos(planet_angle)
        planet_y = planet['a'] * np.sin(planet_angle)
        planet_z = planet['z_offset']
        ax.scatter([planet_x], [planet_y], [planet_z], color=planet['color'], s=150, label=planet_name)
        
        # Add planet label
        ax.text(planet_x, planet_y, planet_z + 0.05*AU, planet_name, color=planet['color'],
                fontweight='bold', fontsize=14, horizontalalignment='center')
    
    # Plot transfer trajectories
    for i, row in inner_df.iterrows():
        trajectory = row['numerical_trajectory']
        planet_name = row['target']
        planet_color = PLANETS[planet_name]['color']
        
        # Add slight vertical offset for better visibility
        trajectory_with_offset = trajectory.copy()
        trajectory_with_offset[:, 2] += PLANETS[planet_name]['z_offset']
        
        ax.plot(trajectory_with_offset[:, 0], trajectory_with_offset[:, 1], trajectory_with_offset[:, 2], 
                label=f"Transfer to {planet_name}", linewidth=3)
        
        # Mark departure and arrival points
        ax.scatter(trajectory_with_offset[0, 0], trajectory_with_offset[0, 1], trajectory_with_offset[0, 2], 
                  color='green', s=100, marker='^')
        ax.scatter(trajectory_with_offset[-1, 0], trajectory_with_offset[-1, 1], trajectory_with_offset[-1, 2], 
                  color='red', s=100, marker='v')
    
    # Set axis limits based on Mars (furthest inner planet)
    max_radius = PLANETS['Mars']['a'] * 1.2
    
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_zlim(-max_radius/4, max_radius/4)
    
    # Convert axis values to AU
    ticks = np.linspace(-max_radius, max_radius, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(np.linspace(-max_radius/4, max_radius/4, 5))
    
    # Format tick labels
    ax.set_xticklabels([f'{t/AU:.1f}' for t in ticks])
    ax.set_yticklabels([f'{t/AU:.1f}' for t in ticks])
    ax.set_zticklabels([f'{t/AU:.1f}' for t in np.linspace(-max_radius/4, max_radius/4, 5)])
    
    ax.set_xlabel('X (AU)', fontweight='bold')
    ax.set_ylabel('Y (AU)', fontweight='bold')
    ax.set_zlabel('Z (AU)', fontweight='bold')
    ax.set_title('Hohmann Transfers from Earth to Inner Planets', fontweight='bold', pad=20)
    
    # Add text annotations for transfer details
    for i, row in inner_df.iterrows():
        planet_name = row['target']
        # Add annotation with key metrics
        info_text = (f"{planet_name}:\n"
                    f"ΔV: {row['delta_v']/1000:.1f} km/s\n"
                    f"Time: {row['tof_days']:.1f} days\n"
                    f"Path: {row['analytical_path_length']/AU:.2f} AU")
        
        # Position the text in a suitable location
        text_x = PLANETS[planet_name]['a'] * np.cos(np.pi/4)
        text_y = PLANETS[planet_name]['a'] * np.sin(np.pi/4)
        text_z = PLANETS[planet_name]['z_offset'] * 3
        
        # Add textbox with information
        ax.text(text_x, text_y, text_z, info_text, color=PLANETS[planet_name]['color'],
               fontweight='bold', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Improve the view angle
    ax.view_init(elev=30, azim=45)
    
    # Create a more manageable legend - separate celestial bodies and transfers
    planet_handles = []
    planet_labels = []
    transfer_handles = []
    transfer_labels = []
    
    handles, labels = ax.get_legend_handles_labels()
    for h, l in zip(handles, labels):
        if "Transfer" in l:
            transfer_handles.append(h)
            transfer_labels.append(l)
        elif "Orbit" not in l:
            planet_handles.append(h)
            planet_labels.append(l)
    
    # Place legends
    ax.legend(planet_handles, planet_labels, loc='upper left', title="Celestial Bodies", frameon=True, framealpha=0.9)
    second_legend = ax.figure.legend(transfer_handles, transfer_labels, loc='upper right', title="Transfer Trajectories", 
                                      frameon=True, framealpha=0.9)
    ax.figure.add_artist(second_legend)
    
    # Handle tight layout warnings safely
    try:
        plt.tight_layout()
    except:
        print("Warning: Tight layout could not be applied to inner planets plot. Continuing anyway.")
    
    plt.savefig('inner_planets_transfers.png', dpi=300, bbox_inches='tight')
    print("Inner planets plot saved as 'inner_planets_transfers.png'")
    
    return fig

def create_comparative_plots(results_df):
    """
    Create bar charts and other comparative visualizations.
    """
    # Set up the plotting style
    sns.set(style="whitegrid", font_scale=1.2)
    
    # Create subplots with more space
    fig, axs = plt.subplots(2, 2, figsize=(24, 18))
    
    # Sort data by distance from Earth
    results_df = results_df.sort_values('a_target')
    
    # Custom colors for better distinction
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # 1. Path Length Comparison
    bars1 = axs[0, 0].bar(results_df['target'], results_df['analytical_path_length']/AU, 
                 color='skyblue', alpha=0.8, label='Analytical', width=0.4)
    bars2 = axs[0, 0].bar(np.arange(len(results_df))+0.4, results_df['numerical_path_length']/AU, 
                 color='navy', alpha=0.6, label='Numerical', width=0.4)
    axs[0, 0].set_title('Transfer Path Length Comparison', fontweight='bold', pad=15)
    axs[0, 0].set_ylabel('Path Length (AU)', fontweight='bold')
    
    # Use linear scale instead of log scale to avoid errors
    axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    axs[0, 0].legend(fontsize=14)
    
    # Add text labels above bars with larger font size
    for i, (_, row) in enumerate(results_df.iterrows()):
        axs[0, 0].text(i, row['analytical_path_length']/AU * 1.1, 
                      f"{row['analytical_path_length']/AU:.1f} AU", 
                      ha='center', va='bottom', rotation=45, fontweight='bold', fontsize=12)
    
    # 2. Time of Flight Comparison
    bars3 = axs[0, 1].bar(results_df['target'], results_df['tof_days'], 
                          color=[colors[i % len(colors)] for i in range(len(results_df))])
    axs[0, 1].set_title('Time of Flight Comparison', fontweight='bold', pad=15)
    axs[0, 1].set_ylabel('Time (days)', fontweight='bold')
    
    # Create a twin axis for years
    ax2 = axs[0, 1].twinx()
    ax2.set_ylabel('Time (years)', fontweight='bold')
    ax2.set_ylim(axs[0, 1].get_ylim()[0]/365.25, axs[0, 1].get_ylim()[1]/365.25)
    
    # Add text labels above bars with clearer formatting
    for i, (_, row) in enumerate(results_df.iterrows()):
        if row['tof_days'] < 365.25:
            # Show days for transfers less than a year
            text = f"{row['tof_days']:.1f} days"
        else:
            # Show years for longer transfers
            text = f"{row['tof_days']:.1f} days\n({row['tof_days']/365.25:.1f} yrs)"
            
        axs[0, 1].text(i, row['tof_days'] * 1.05, text, 
                      ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Use linear scale for all plots
    axs[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. Delta-V Comparison - use horizontal bars for better readability
    # Sort by delta-V for this chart
    dv_df = results_df.sort_values('delta_v')
    
    bars4 = axs[1, 0].barh(dv_df['target'], dv_df['departure_dv']/1000, 
                         color='orange', alpha=0.8, label='Departure ΔV', height=0.6)
    bars5 = axs[1, 0].barh(dv_df['target'], dv_df['arrival_dv']/1000, 
                         left=dv_df['departure_dv']/1000, 
                         color='red', alpha=0.8, label='Arrival ΔV', height=0.6)
    axs[1, 0].set_title('Delta-V Requirements', fontweight='bold', pad=15)
    axs[1, 0].set_xlabel('Delta-V (km/s)', fontweight='bold')
    axs[1, 0].legend(fontsize=14, loc='upper right')
    axs[1, 0].grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add text labels for total delta-V
    for i, (_, row) in enumerate(dv_df.iterrows()):
        total_dv = row['delta_v']/1000
        axs[1, 0].text(total_dv + 0.5, i, 
                      f"{total_dv:.1f} km/s", 
                      va='center', fontweight='bold', fontsize=12)
    
    # 4. Relationship plot - use normalized values for better comparison
    # Create normalized versions of our data
    norm_data = results_df.copy()
    norm_data['norm_distance'] = norm_data['a_target'] / norm_data['a_target'].max()
    norm_data['norm_time'] = norm_data['tof_days'] / norm_data['tof_days'].max()
    norm_data['norm_path'] = norm_data['analytical_path_length'] / norm_data['analytical_path_length'].max()
    norm_data['norm_dv'] = norm_data['delta_v'] / norm_data['delta_v'].max()
    
    # Set width of each group of bars
    width = 0.2
    x = np.arange(len(norm_data))
    
    # Create a grouped bar chart showing normalized values
    axs[1, 1].bar(x - width*1.5, norm_data['norm_distance'], width, label='Orbital Distance', color='steelblue')
    axs[1, 1].bar(x - width/2, norm_data['norm_path'], width, label='Path Length', color='forestgreen')
    axs[1, 1].bar(x + width/2, norm_data['norm_time'], width, label='Time of Flight', color='darkorange') 
    axs[1, 1].bar(x + width*1.5, norm_data['norm_dv'], width, label='Delta-V', color='firebrick')
    
    # Configure the plot
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(norm_data['target'])
    axs[1, 1].set_title('Normalized Comparison of Key Metrics', fontweight='bold', pad=15)
    axs[1, 1].set_ylabel('Normalized Value (ratio to maximum)', fontweight='bold')
    axs[1, 1].legend(fontsize=12)
    axs[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add correlation coefficients as text
    try:
        from scipy import stats
        corr_dist_time = stats.pearsonr(results_df['a_target'], results_df['tof_days'])[0]
        corr_dist_dv = stats.pearsonr(results_df['a_target'], results_df['delta_v'])[0]
        corr_text = (f"Correlations:\n"
                     f"Distance-Time: {corr_dist_time:.2f}\n"
                     f"Distance-ΔV: {corr_dist_dv:.2f}")
        axs[1, 1].text(0.02, 0.95, corr_text, transform=axs[1, 1].transAxes,
                     fontsize=14, fontweight='bold', 
                     bbox=dict(facecolor='white', alpha=0.8))
    except Exception as e:
        print(f"Could not add correlation data: {e}")
    
    # Handle tight layout warnings safely
    try:
        plt.tight_layout(pad=3.0)
    except Exception as e:
        print(f"Warning: Tight layout could not be applied to comparison plots. {str(e)}")
    
    # Make sure the figure is saved even if tight_layout fails
    try:
        plt.savefig('hohmann_transfers_comparison.png', dpi=300, bbox_inches='tight')
        print("Comparison plots saved as 'hohmann_transfers_comparison.png'")
    except Exception as e:
        print(f"Error saving comparison plots: {str(e)}")
        # Try with a more basic save approach
        try:
            plt.savefig('hohmann_transfers_comparison.png', dpi=300)
            print("Comparison plots saved with basic settings as 'hohmann_transfers_comparison.png'")
        except Exception as e2:
            print(f"Failed to save comparison plots: {str(e2)}")
    
    return fig

def analyze_and_report(results_df):
    """
    Analyze the results and generate a summary report.
    """
    # Find minimum and maximum values
    min_path = results_df.loc[results_df['analytical_path_length'].idxmin()]
    max_path = results_df.loc[results_df['analytical_path_length'].idxmax()]
    
    min_time = results_df.loc[results_df['tof_days'].idxmin()]
    max_time = results_df.loc[results_df['tof_days'].idxmax()]
    
    min_dv = results_df.loc[results_df['delta_v'].idxmin()]
    max_dv = results_df.loc[results_df['delta_v'].idxmax()]
    
    # Print summary
    print("\n===== INTERPLANETARY HOHMANN TRANSFERS SUMMARY =====")
    print(f"\nTotal planets analyzed: {len(results_df)}")
    
    print("\nPath Length Analysis:")
    print(f"  Shortest path: {min_path['target']} ({min_path['analytical_path_length']/AU:.2f} AU)")
    print(f"  Longest path: {max_path['target']} ({max_path['analytical_path_length']/AU:.2f} AU)")
    print(f"  Ratio of longest to shortest: {max_path['analytical_path_length']/min_path['analytical_path_length']:.2f}")
    
    print("\nTime of Flight Analysis:")
    print(f"  Shortest time: {min_time['target']} ({min_time['tof_days']:.2f} days, {min_time['tof_days']/365.25:.2f} years)")
    print(f"  Longest time: {max_time['target']} ({max_time['tof_days']:.2f} days, {max_time['tof_days']/365.25:.2f} years)")
    print(f"  Ratio of longest to shortest: {max_time['tof_days']/min_time['tof_days']:.2f}")
    
    print("\nDelta-V Analysis:")
    print(f"  Lowest delta-V: {min_dv['target']} ({min_dv['delta_v']/1000:.2f} km/s)")
    print(f"  Highest delta-V: {max_dv['target']} ({max_dv['delta_v']/1000:.2f} km/s)")
    print(f"  Ratio of highest to lowest: {max_dv['delta_v']/min_dv['delta_v']:.2f}")
    
    print("\nDetailed Planetary Data:")
    print(f"{'Planet':<10} {'Path (AU)':<12} {'Time (days)':<15} {'Time (years)':<15} {'ΔV (km/s)':<12}")
    print("-" * 70)
    
    for _, row in results_df.sort_values('a_target').iterrows():
        print(f"{row['target']:<10} {row['analytical_path_length']/AU:<12.2f} {row['tof_days']:<15.2f} {row['tof_days']/365.25:<15.2f} {row['delta_v']/1000:<12.2f}")
    
    # Generate correlation matrix
    print("\nCorrelation Analysis:")
    corr_data = results_df[['a_target', 'analytical_path_length', 'numerical_path_length', 'tof_days', 'delta_v']]
    corr_matrix = corr_data.corr()
    
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    # Create a data table for the report
    summary_table = results_df[['target', 'analytical_path_length', 'tof_days', 'delta_v']]
    summary_table = summary_table.rename(columns={
        'target': 'Planet',
        'analytical_path_length': 'Path Length (m)',
        'tof_days': 'Time of Flight (days)',
        'delta_v': 'Delta-V (m/s)'
    })
    
    # Convert to AU, years, km/s for readability
    summary_table['Path Length (AU)'] = summary_table['Path Length (m)'] / AU
    summary_table['Time of Flight (years)'] = summary_table['Time of Flight (days)'] / 365.25
    summary_table['Delta-V (km/s)'] = summary_table['Delta-V (m/s)'] / 1000
    
    # Reorder and select columns
    final_table = summary_table[['Planet', 'Path Length (AU)', 'Time of Flight (days)', 
                                'Time of Flight (years)', 'Delta-V (km/s)']]
    
    # Save to CSV
    final_table.to_csv('hohmann_transfers_summary.csv', index=False)
    print("\nDetailed results saved to 'hohmann_transfers_summary.csv'")
    
    return final_table

def main():
    """Main function to run the entire analysis."""
    print("Interplanetary Hohmann Transfer Analysis")
    print("======================================")
    
    try:
        # Calculate transfer parameters
        print("Calculating transfer parameters for all planets...")
        results_df = calculate_all_transfers()
        
        # Create visualizations
        print("\nGenerating visualizations...")
        plot_transfer_trajectories(results_df)
        create_inner_planets_plot(results_df)
        create_comparative_plots(results_df)
        
        # Analyze and report results
        print("\nAnalyzing results...")
        summary_table = analyze_and_report(results_df)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 