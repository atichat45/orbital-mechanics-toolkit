"""
Orbit Plotting Module

This module provides functions for visualizing orbits, trajectories, and related
orbital mechanics concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from datetime import datetime, timedelta

from ..utils.constants import AU, SUN_RADIUS, PLANETS
from ..core.orbital_elements import keplerian_to_cartesian

# Configure default plotting parameters for better readability
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (10, 8),
    'savefig.dpi': 300
})

def plot_orbit_3d(ax, semi_major_axis, eccentricity=0, inclination=0, 
                 raan=0, arg_periapsis=0, color='b', label=None, linestyle='-', 
                 num_points=100, linewidth=2, alpha=0.7, mu=1.32712440018e20):
    """
    Plot a 3D orbit based on Keplerian elements.
    
    Args:
        ax: Matplotlib 3D axis
        semi_major_axis: Semi-major axis (m)
        eccentricity: Eccentricity (unitless)
        inclination: Inclination (rad)
        raan: Right ascension of ascending node (rad)
        arg_periapsis: Argument of periapsis (rad)
        color: Orbit color
        label: Orbit label for legend
        linestyle: Line style for orbit
        num_points: Number of points to plot
        linewidth: Width of orbit line
        alpha: Transparency of orbit line
        mu: Gravitational parameter (default is Sun's value in m^3/s^2)
    
    Returns:
        Line object for the orbit
    """
    # Generate true anomaly values
    true_anomaly = np.linspace(0, 2*np.pi, num_points)
    
    # Initialize arrays for position
    positions = np.zeros((num_points, 3))
    
    # Generate position vectors for each true anomaly
    for i, nu in enumerate(true_anomaly):
        # Create orbital elements dictionary
        elements = {
            'a': semi_major_axis,
            'e': eccentricity,
            'i': inclination,
            'Omega': raan,
            'omega': arg_periapsis,
            'nu': nu
        }
        
        # Convert to Cartesian coordinates using the provided mu
        r, _ = keplerian_to_cartesian(elements, mu)
        positions[i] = r
    
    # Plot the orbit
    orbit_line = ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                        color=color, label=label, linestyle=linestyle, 
                        linewidth=linewidth, alpha=alpha)[0]
    
    return orbit_line

def plot_trajectory_3d(ax, positions, color='g', label=None, linestyle='-', 
                      linewidth=2, alpha=0.7, show_points=True, point_size=50):
    """
    Plot a 3D trajectory from a series of positions.
    
    Args:
        ax: Matplotlib 3D axis
        positions: Array of position vectors Nx3 (x, y, z)
        color: Trajectory color
        label: Trajectory label for legend
        linestyle: Line style for trajectory
        linewidth: Width of trajectory line
        alpha: Transparency of trajectory line
        show_points: If True, show start and end points
        point_size: Size of start/end points
    
    Returns:
        Line object for the trajectory
    """
    # Convert positions to numpy array if not already
    positions = np.array(positions)
    
    # Plot the trajectory line
    trajectory_line = ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                             color=color, label=label, linestyle=linestyle, 
                             linewidth=linewidth, alpha=alpha)[0]
    
    # Show start and end points if requested
    if show_points and len(positions) > 1:
        # Start point
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                  color=color, s=point_size, alpha=1.0)
        
        # End point
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                  color=color, s=point_size, alpha=1.0)
    
    return trajectory_line

def setup_solar_system_plot(ax, max_radius=None, title="Solar System Plot"):
    """
    Configure a 3D axis for a solar system plot.
    
    Args:
        ax: Matplotlib 3D axis
        max_radius: Maximum radius for axis limits (default: None, auto-calculated)
        title: Plot title
    
    Returns:
        The configured axis
    """
    # Plot the Sun
    ax.scatter([0], [0], [0], color='yellow', s=200, label='Sun')
    
    # Add a light yellow sphere for the Sun (not to scale, just for visualization)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    r_sun_plot = SUN_RADIUS * 5 / AU  # Increased for visibility
    x = r_sun_plot * np.cos(u) * np.sin(v)
    y = r_sun_plot * np.sin(u) * np.sin(v)
    z = r_sun_plot * np.cos(v)
    ax.plot_surface(x, y, z, color='yellow', alpha=0.2)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Set plot limits based on max_radius
    if max_radius is None:
        max_radius = max(planet['semi_major_axis'] for planet in PLANETS.values()) / AU
    
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_zlim(-max_radius * 0.5, max_radius * 0.5)  # Less in z-direction for better visibility
    
    # Set labels and title
    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    ax.set_title(title)
    
    # Add grid for better orientation
    ax.grid(True, alpha=0.3)
    
    # Improve view angle
    ax.view_init(elev=30, azim=45)
    
    return ax

def plot_planet_orbit(ax, planet_name, position=None, show_label=True, 
                     z_offset=None, **kwargs):
    """
    Plot a planet's orbit around the Sun.
    
    Args:
        ax: Matplotlib 3D axis
        planet_name: Name of the planet to plot
        position: Optional true anomaly position to show the planet (rad)
        show_label: If True, show the planet label
        z_offset: Optional z-offset for the orbit (for visualization purposes)
        **kwargs: Additional arguments passed to plot_orbit_3d
    
    Returns:
        Tuple of (orbit_line, planet_point) or just orbit_line if position is None
    """
    # Check if the planet exists
    if planet_name not in PLANETS:
        raise ValueError(f"Unknown planet: {planet_name}")
    
    planet = PLANETS[planet_name]
    
    # Get default color for the planet
    color = kwargs.pop('color', planet['color'])
    
    # Create label
    label = f"{planet_name}" if show_label else None
    
    # Convert orbital elements to AU for plotting
    a = planet['semi_major_axis'] / AU
    e = planet['eccentricity']
    
    # Default inclination, RAAN, and arg of periapsis are zero for simple solar system model
    i = kwargs.pop('inclination', 0)
    raan = kwargs.pop('raan', 0)
    arg_periapsis = kwargs.pop('arg_periapsis', 0)
    
    # Plot the orbit
    orbit_line = plot_orbit_3d(
        ax, 
        semi_major_axis=a, 
        eccentricity=e,
        inclination=i,
        raan=raan,
        arg_periapsis=arg_periapsis,
        color=color,
        label=label,
        mu=1.0,  # For plotting purposes, we've already scaled by AU
        **kwargs
    )
    
    # If a position is provided, show the planet at that position
    if position is not None:
        # Create orbital elements dictionary
        elements = {
            'a': a,  # Already in AU
            'e': e,
            'i': i,
            'Omega': raan,
            'omega': arg_periapsis,
            'nu': position
        }
        
        # Convert to Cartesian coordinates - use mu=1 since we're working in AU
        r, _ = keplerian_to_cartesian(elements, 1.0)
        
        # Apply z-offset if provided
        if z_offset is not None:
            r[2] += z_offset
        
        # Plot the planet
        planet_point = ax.scatter(
            r[0], r[1], r[2], 
            color=color, 
            s=30 * (planet['radius'] / PLANETS['Earth']['radius'])**0.5,  # Scale by sqrt of radius ratio
            label=None,
            alpha=1.0
        )
        
        return orbit_line, planet_point
    
    return orbit_line

def plot_hohmann_transfer(ax, r1, r2, departure_angle=0, clockwise=False, color='r',
                         label="Transfer Orbit", **kwargs):
    """
    Visualize a Hohmann transfer orbit between two circular orbits.
    
    Args:
        ax: Matplotlib 3D axis
        r1: Radius of departure circular orbit (AU)
        r2: Radius of arrival circular orbit (AU)
        departure_angle: Angle of departure (rad)
        clockwise: If True, plot clockwise transfer
        color: Color of transfer orbit
        label: Label for the transfer orbit
        **kwargs: Additional arguments passed to plot_orbit_3d
    
    Returns:
        Line object for the transfer orbit
    """
    # Semi-major axis of the transfer orbit
    a_transfer = (r1 + r2) / 2
    
    # Eccentricity of the transfer orbit
    e_transfer = abs(r2 - r1) / (r2 + r1)
    
    # Adjust for orientation
    if r1 > r2:
        # Going inward
        arg_periapsis = departure_angle + np.pi if clockwise else departure_angle
    else:
        # Going outward
        arg_periapsis = departure_angle if clockwise else departure_angle + np.pi
    
    # Plot transfer orbit (half an ellipse)
    transfer_line = plot_orbit_3d(
        ax, 
        semi_major_axis=a_transfer, 
        eccentricity=e_transfer,
        arg_periapsis=arg_periapsis,
        color=color,
        label=label,
        num_points=100,
        **kwargs
    )
    
    return transfer_line

def create_orbit_animation(positions, interval=50, blit=True, save_path=None):
    """
    Create an animation of an orbital trajectory.
    
    Args:
        positions: Array of position vectors Nx3 (x, y, z)
        interval: Time interval between frames (ms)
        blit: Whether to use blitting for improved performance
        save_path: Path to save the animation (if None, animation is not saved)
    
    Returns:
        Animation object
    """
    positions = np.array(positions)
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Orbital Trajectory Animation')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Initialize plot elements
    trajectory_line, = ax.plot([], [], [], 'b-', label='Full Trajectory', alpha=0.3)
    current_point = ax.scatter([], [], [], color='r', s=50, label='Current Position')
    
    # Set axis limits based on the full trajectory
    max_limit = np.max(np.abs(positions)) * 1.1
    ax.set_xlim(-max_limit, max_limit)
    ax.set_ylim(-max_limit, max_limit)
    ax.set_zlim(-max_limit, max_limit)
    
    # Plot the full trajectory for reference
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', alpha=0.3)
    
    # Animation init function
    def init():
        trajectory_line.set_data([], [])
        trajectory_line.set_3d_properties([])
        current_point._offsets3d = ([], [], [])
        return trajectory_line, current_point
    
    # Animation update function
    def animate(i):
        # Plot the trajectory up to the current point
        trajectory_line.set_data(positions[:i+1, 0], positions[:i+1, 1])
        trajectory_line.set_3d_properties(positions[:i+1, 2])
        
        # Update the current point
        current_point._offsets3d = ([positions[i, 0]], [positions[i, 1]], [positions[i, 2]])
        
        return trajectory_line, current_point
    
    # Create the animation
    anim = FuncAnimation(
        fig, animate, frames=len(positions),
        init_func=init, interval=interval, blit=blit
    )
    
    # Save the animation if a path is provided
    if save_path is not None:
        anim.save(save_path, writer='pillow', fps=30)
    
    return anim

def plot_porkchop(ax, data, title, c3_levels=None, dv_arrival_levels=None):
    """
    Create a porkchop plot for launch window analysis.
    
    Args:
        ax: Matplotlib axis
        data: Dictionary with porkchop data (departure_dates, tof_days, c3, delta_v)
        title: Plot title
        c3_levels: List of contour levels for C3
        dv_arrival_levels: List of contour levels for arrival delta-V
        
    Returns:
        Dictionary with contour objects
    """
    # Extract data
    departure_dates = data['departure_dates']
    tof_days = data['tof_days']
    c3 = data['c3']
    delta_v = data['delta_v']
    
    # Create mesh grids for contour plot
    departure_grid, tof_grid = np.meshgrid(
        mdates.date2num(departure_dates),
        tof_days
    )
    
    # Set default contour levels if not provided
    if c3_levels is None:
        c3_min = np.nanmin(c3)
        c3_max = np.nanmax(c3)
        c3_levels = np.linspace(c3_min, c3_max, 15)
    
    if dv_arrival_levels is None:
        dv_min = np.nanmin(delta_v)
        dv_max = np.nanmax(delta_v)
        dv_arrival_levels = np.linspace(dv_min, dv_max, 15)
    
    # Create contour plots
    c3_contour = ax.contour(
        departure_grid, 
        tof_grid, 
        c3.T,  # Transpose to match the grid
        levels=c3_levels,
        colors='blue',
        linestyles='solid',
        linewidths=1.5,
        alpha=0.7
    )
    
    dv_contour = ax.contour(
        departure_grid,
        tof_grid,
        delta_v.T,  # Transpose to match the grid
        levels=dv_arrival_levels,
        colors='red',
        linestyles='dashed',
        linewidths=1.5,
        alpha=0.7
    )
    
    # Add contour labels
    ax.clabel(c3_contour, inline=True, fontsize=8, fmt='%.1f', colors='blue')
    ax.clabel(dv_contour, inline=True, fontsize=8, fmt='%.1f', colors='red')
    
    # Add colorbar for C3
    plt.colorbar(c3_contour, ax=ax, pad=0.05, label='C3 (km²/s²)')
    
    # Configure axis
    ax.set_xlabel('Departure Date')
    ax.set_ylabel('Time of Flight (days)')
    ax.set_title(title)
    
    # Format x-axis dates
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    c3_line = plt.Line2D([0], [0], color='blue', linestyle='solid', label='C3 (km²/s²)')
    dv_line = plt.Line2D([0], [0], color='red', linestyle='dashed', label='ΔV (km/s)')
    ax.legend(handles=[c3_line, dv_line], loc='upper right')
    
    return {
        'c3_contour': c3_contour,
        'dv_contour': dv_contour
    } 