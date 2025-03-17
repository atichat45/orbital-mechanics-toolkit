#!/usr/bin/env python3
"""
Mars Porkchop Plot Generator

This script calculates and visualizes a porkchop plot for Earth-to-Mars transfer
opportunities, showing delta-V requirements for different departure and arrival dates.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import datetime as dt
from scipy.optimize import minimize_scalar

# Constants
G = 6.67430e-11  # Gravitational constant, m^3/kg/s^2
M_SUN = 1.989e30  # Solar mass, kg
AU = 149.6e9  # Astronomical unit, m
DAY = 86400  # Seconds in a day

# Convert gravitational parameter to m^3/s^2
MU_SUN = G * M_SUN

# Earth constants
M_EARTH = 5.972e24  # Earth mass, kg
R_EARTH = 6.371e6   # Earth radius, m
MU_EARTH = G * M_EARTH

# Mars constants
M_MARS = 6.39e23  # Mars mass, kg
R_MARS = 3.389e6  # Mars radius, m
MU_MARS = G * M_MARS

# Low Earth Orbit (LEO) altitude
LEO_ALTITUDE = 300e3  # m
LEO_RADIUS = R_EARTH + LEO_ALTITUDE
LEO_VELOCITY = np.sqrt(MU_EARTH / LEO_RADIUS)

# Mars orbit altitude
MARS_ORBIT_ALTITUDE = 400e3  # m
MARS_ORBIT_RADIUS = R_MARS + MARS_ORBIT_ALTITUDE
MARS_ORBIT_VELOCITY = np.sqrt(MU_MARS / MARS_ORBIT_RADIUS)

# Orbital parameters - simplified mean elements
EARTH_ORBIT = {
    'a': 1.000373 * AU,  # Semi-major axis
    'e': 0.0167086,      # Eccentricity
    'i': np.radians(0.00005),  # Inclination
    'raan': np.radians(180.0),  # Right ascension of ascending node
    'argp': np.radians(102.94719),  # Argument of periapsis
    'period': 365.256363004 * DAY,  # Orbital period
    'epoch': dt.datetime(2000, 1, 1, 12, 0, 0)  # J2000 epoch
}

MARS_ORBIT = {
    'a': 1.52366231 * AU,  # Semi-major axis
    'e': 0.09341233,      # Eccentricity
    'i': np.radians(1.85061),  # Inclination
    'raan': np.radians(49.57854),  # Right ascension of ascending node
    'argp': np.radians(336.04084),  # Argument of periapsis
    'period': 686.9800 * DAY,  # Orbital period
    'epoch': dt.datetime(2000, 1, 1, 12, 0, 0)  # J2000 epoch
}

def datetime_to_seconds(date, epoch=EARTH_ORBIT['epoch']):
    """Convert datetime to seconds since epoch."""
    return (date - epoch).total_seconds()

def get_mean_anomaly(orbit, time_seconds):
    """
    Calculate mean anomaly at specified time.
    
    Args:
        orbit: Dictionary with orbital elements
        time_seconds: Time in seconds since epoch
    
    Returns:
        Mean anomaly in radians
    """
    # Mean motion
    n = 2 * np.pi / orbit['period']
    
    # Mean anomaly at epoch (simplified)
    M0 = 0
    
    # Mean anomaly at requested time
    M = M0 + n * time_seconds
    
    # Normalize to [0, 2π]
    return M % (2 * np.pi)

def solve_kepler(M, e, max_iter=100, tol=1e-8):
    """
    Solve Kepler's equation for eccentric anomaly.
    
    Args:
        M: Mean anomaly (rad)
        e: Eccentricity
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
    
    Returns:
        Eccentric anomaly (rad)
    """
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
    
    # Return best guess if not converged
    return E

def get_position_and_velocity(orbit, time_seconds):
    """
    Calculate position and velocity vectors at specified time.
    
    Args:
        orbit: Dictionary with orbital elements
        time_seconds: Time in seconds since epoch
    
    Returns:
        Tuple (position, velocity) with 3D vectors in m and m/s
    """
    # Extract orbital elements
    a = orbit['a']
    e = orbit['e']
    i = orbit['i']
    raan = orbit['raan']
    argp = orbit['argp']
    
    # Calculate mean anomaly
    M = get_mean_anomaly(orbit, time_seconds)
    
    # Solve Kepler's equation for eccentric anomaly
    E = solve_kepler(M, e)
    
    # Calculate true anomaly
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    
    # Calculate distance from focal point
    r = a * (1 - e * np.cos(E))
    
    # Calculate position in orbital plane
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    z_orb = 0
    
    # Calculate velocity in orbital plane
    p = a * (1 - e**2)
    h = np.sqrt(MU_SUN * p)
    
    vx_orb = -(h / r) * np.sin(nu)
    vy_orb = (h / r) * (e + np.cos(nu))
    vz_orb = 0
    
    # Rotation matrices
    R3_raan = np.array([
        [np.cos(raan), -np.sin(raan), 0],
        [np.sin(raan), np.cos(raan), 0],
        [0, 0, 1]
    ])
    
    R1_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]
    ])
    
    R3_argp = np.array([
        [np.cos(argp), -np.sin(argp), 0],
        [np.sin(argp), np.cos(argp), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation
    R = np.matmul(R3_raan, np.matmul(R1_i, R3_argp))
    
    # Transform position and velocity to reference frame
    pos_orb = np.array([x_orb, y_orb, z_orb])
    vel_orb = np.array([vx_orb, vy_orb, vz_orb])
    
    pos = np.matmul(R, pos_orb)
    vel = np.matmul(R, vel_orb)
    
    return pos, vel

def lambert_solve(r1, r2, tof, mu, direction=1):
    """
    Solve Lambert's problem using a simplified method.
    
    Args:
        r1: Initial position vector (m)
        r2: Final position vector (m)
        tof: Time of flight (seconds)
        mu: Gravitational parameter (m^3/s^2)
        direction: 1 for prograde, -1 for retrograde
    
    Returns:
        Tuple (v1, v2) of initial and final velocity vectors
    """
    # Magnitudes
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)
    
    # Unit vectors
    ir1 = r1 / r1_mag
    ir2 = r2 / r2_mag
    
    # Cross product of unit vectors
    crossp = np.cross(ir1, ir2)
    crossp_mag = np.linalg.norm(crossp)
    
    # Calculate the sine and cosine of the transfer angle
    cos_dnu = np.dot(ir1, ir2)
    
    # Determine transfer angle
    if direction == 1:  # Prograde
        if crossp_mag < 1e-10:  # Nearly coplanar
            if cos_dnu > 0:
                dnu = np.arccos(cos_dnu)  # 0 to π
            else:
                dnu = 2 * np.pi - np.arccos(cos_dnu)  # π to 2π
        else:
            dnu = np.arctan2(crossp_mag, cos_dnu)
            if dnu < 0:
                dnu += 2 * np.pi
    else:  # Retrograde
        if crossp_mag < 1e-10:
            if cos_dnu > 0:
                dnu = 2 * np.pi - np.arccos(cos_dnu)
            else:
                dnu = np.arccos(cos_dnu)
        else:
            dnu = np.arctan2(-crossp_mag, cos_dnu)
            if dnu < 0:
                dnu += 2 * np.pi
    
    # Chord distance
    c = np.sqrt(r1_mag**2 + r2_mag**2 - 2 * r1_mag * r2_mag * np.cos(dnu))
    
    # Semi-perimeter
    s = (r1_mag + r2_mag + c) / 2
    
    # Calculate minimum energy orbit's semi-major axis (parabolic limit)
    a_min = s / 2
    
    # Universal variable formulation - start with the value that gives a_min
    x0 = np.sqrt(r1_mag * r2_mag) * np.sin(dnu/2) / np.sqrt(s)
    
    # Function to calculate time of flight for a given value of x
    def get_tof_from_x(x):
        if abs(x) < 1e-10:  # Parabolic (minimum energy)
            a = float('inf')
            y = s
        else:
            a = s / (1 - x**2)  # Semi-major axis
            y = r1_mag + r2_mag + c * x * np.sqrt(1 - x**2) / np.sqrt(s)
        
        if x < 1:  # Elliptical
            beta = 2 * np.arcsin(np.sqrt((s - c) / (2 * a)))
            if dnu > np.pi:
                beta = -beta
            
            alfa = 2 * np.arcsin(np.sqrt(s / (2 * a)))
            t_ellip = np.sqrt(a**3 / mu) * ((alfa - np.sin(alfa)) - (beta - np.sin(beta)))
            return t_ellip
        else:  # Hyperbolic
            beta = 2 * np.arcsinh(np.sqrt((s - c) / (-2 * a)))
            if dnu > np.pi:
                beta = -beta
            
            alfa = 2 * np.arcsinh(np.sqrt(s / (-2 * a)))
            t_hyper = np.sqrt((-a)**3 / mu) * ((np.sinh(alfa) - alfa) - (np.sinh(beta) - beta))
            return t_hyper
    
    # Function to minimize to find the correct orbit
    def tof_diff(x):
        return abs(get_tof_from_x(x) - tof)
    
    # Try to find best value of x using the Brent method
    try:
        # For elliptical transfers
        res = minimize_scalar(tof_diff, bracket=[-0.9, x0, 0.9], method='brent')
        x = res.x
    except:
        # Fallback to a simpler approach
        # Just use minimum energy transfer (parabolic)
        x = 0
    
    # Once x is found, calculate semi-major axis
    if abs(x) < 1e-10:
        a = float('inf')  # Parabolic orbit
        p = 2 * a_min     # p = 2a for parabolic orbit
    else:
        a = s / (1 - x**2)
        # Handle very large a
        if abs(a) > 1e20:
            a = 1e20 * np.sign(a)
        
        # Calculate semi-latus rectum p
        if a > 0:  # Elliptical
            e = np.sqrt(1 - (p := (4 * a * a_min * (1 - x**2)) / s**2))
        else:      # Hyperbolic
            e = np.sqrt(1 + (p := (4 * a * a_min * (x**2 - 1)) / s**2))
    
    # Calculate velocities using the Lagrange coefficients
    # Determine the normal vector to the transfer plane
    if crossp_mag < 1e-10:
        # If nearly coplanar, pick any perpendicular direction
        h_hat = np.array([0, 0, 1]) if abs(ir1[2]) < 0.9 else np.array([1, 0, 0])
        h_hat = h_hat - ir1 * np.dot(h_hat, ir1)
        h_hat = h_hat / np.linalg.norm(h_hat)
    else:
        h_hat = crossp / crossp_mag
    
    # Time derivatives of Lagrange coefficients
    if abs(dnu - np.pi) < 1e-10:  # 180-degree transfer
        # For 180-degree transfers, use Hohmann-like approximation
        v1_p = np.sqrt(mu * (2/r1_mag - 1/a))
        v2_p = np.sqrt(mu * (2/r2_mag - 1/a))
        
        # Get transfer plane velocity directions
        it1 = np.cross(h_hat, ir1)
        it2 = -np.cross(h_hat, ir2)
        
        # Final velocities
        v1 = v1_p * it1
        v2 = v2_p * it2
    else:
        # Normal case: calculate using Lagrange coefficients
        # Chord vector
        c_vec = r2 - r1
        
        # Calculate Lagrange coefficients
        f = 1 - r2_mag/p * (1 - np.cos(dnu))
        g = r1_mag * r2_mag * np.sin(dnu) / np.sqrt(mu * p)
        g_dot = 1 - r1_mag/p * (1 - np.cos(dnu))
        
        # Calculate velocities
        v1 = (r2 - f * r1) / g
        v2 = (g_dot * r2 - r1) / g
    
    return v1, v2

def calculate_interplanetary_transfer(departure_date, arrival_date):
    """
    Calculate interplanetary transfer from Earth to Mars.
    
    Args:
        departure_date: Departure datetime
        arrival_date: Arrival datetime
    
    Returns:
        Dictionary with transfer parameters
    """
    # Convert dates to seconds since epoch
    departure_seconds = datetime_to_seconds(departure_date)
    arrival_seconds = datetime_to_seconds(arrival_date)
    
    # Calculate time of flight
    tof = arrival_seconds - departure_seconds
    
    # Get positions and velocities at departure and arrival
    earth_pos, earth_vel = get_position_and_velocity(EARTH_ORBIT, departure_seconds)
    mars_pos, mars_vel = get_position_and_velocity(MARS_ORBIT, arrival_seconds)
    
    try:
        # Calculate transfer orbit using Lambert's solution
        v1, v2 = lambert_solve(earth_pos, mars_pos, tof, MU_SUN)
        
        # Calculate delta-v at departure and arrival
        delta_v1 = np.linalg.norm(v1 - earth_vel)
        delta_v2 = np.linalg.norm(mars_vel - v2)
        
        # Calculate Earth departure parameters
        v_inf_earth = delta_v1
        c3 = v_inf_earth**2
        v_departure = np.sqrt(v_inf_earth**2 + 2 * MU_EARTH / LEO_RADIUS)
        delta_v_departure = v_departure - LEO_VELOCITY
        
        # Calculate Mars arrival parameters
        v_inf_mars = delta_v2
        v_approach = np.sqrt(v_inf_mars**2 + 2 * MU_MARS / MARS_ORBIT_RADIUS)
        delta_v_arrival = v_approach - MARS_ORBIT_VELOCITY
        
        # Calculate total mission delta-v
        total_delta_v = delta_v_departure + delta_v_arrival
        
        return {
            'departure_date': departure_date,
            'arrival_date': arrival_date,
            'tof': tof / DAY,  # days
            'delta_v1': delta_v1,
            'delta_v2': delta_v2,
            'delta_v_departure': delta_v_departure,
            'delta_v_arrival': delta_v_arrival,
            'total_delta_v': total_delta_v,
            'c3': c3,
            'earth_pos': earth_pos,
            'mars_pos': mars_pos,
            'v1': v1,
            'v2': v2,
            'success': True
        }
    except Exception as e:
        # Return failure if Lambert solver fails
        return {
            'departure_date': departure_date,
            'arrival_date': arrival_date,
            'success': False,
            'error': str(e)
        }

def calculate_porkchop_data(start_date, end_date, min_tof, max_tof, step_days=10, tof_step_days=10):
    """
    Calculate data for porkchop plot.
    
    Args:
        start_date: Start date for departure window
        end_date: End date for departure window
        min_tof: Minimum time of flight in days
        max_tof: Maximum time of flight in days
        step_days: Step size for departure dates in days
        tof_step_days: Step size for time of flight in days
    
    Returns:
        Tuple (departure_dates, arrival_dates, delta_v_grid, c3_grid)
    """
    # Generate departure date range
    num_departure_days = (end_date - start_date).days
    num_departure_points = num_departure_days // step_days + 1
    departure_range = [start_date + dt.timedelta(days=i*step_days) for i in range(num_departure_points)]
    
    # Generate time of flight range
    num_tof_points = (max_tof - min_tof) // tof_step_days + 1
    tof_range = [min_tof + i*tof_step_days for i in range(num_tof_points)]
    
    # Initialize arrays for results
    num_deps = len(departure_range)
    num_tofs = len(tof_range)
    
    delta_v_grid = np.zeros((num_deps, num_tofs))
    c3_grid = np.zeros((num_deps, num_tofs))
    
    # Create arrival dates grid
    arrival_dates = [[departure_date + dt.timedelta(days=tof) 
                      for tof in tof_range] 
                     for departure_date in departure_range]
    
    # Calculate transfer for each combination
    for i, departure_date in enumerate(departure_range):
        print(f"Processing departure date {i+1}/{num_deps}: {departure_date.strftime('%Y-%m-%d')}")
        for j, tof in enumerate(tof_range):
            arrival_date = departure_date + dt.timedelta(days=tof)
            
            # Calculate transfer
            result = calculate_interplanetary_transfer(departure_date, arrival_date)
            
            # Store results if successful
            if result['success']:
                delta_v_grid[i, j] = result['total_delta_v'] / 1000  # km/s
                c3_grid[i, j] = result['c3'] / 1e6  # km²/s²
            else:
                delta_v_grid[i, j] = np.nan
                c3_grid[i, j] = np.nan
    
    return departure_range, arrival_dates, delta_v_grid, c3_grid

def plot_porkchop(departure_dates, arrival_dates, delta_v_grid, c3_grid, filename="mars_porkchop.png"):
    """
    Create a porkchop plot.
    
    Args:
        departure_dates: Array of departure dates
        arrival_dates: 2D array of arrival dates
        delta_v_grid: 2D array of total delta-v values
        c3_grid: 2D array of C3 values
        filename: Output filename
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Find minimum delta-v and its location
    min_dv = np.nanmin(delta_v_grid)
    min_idx = np.unravel_index(np.nanargmin(delta_v_grid), delta_v_grid.shape)
    min_departure = departure_dates[min_idx[0]]
    min_arrival = arrival_dates[min_idx[0], min_idx[1]]
    min_tof = (min_arrival - min_departure).days
    
    # Delta-V Contour Plot
    vmin = np.nanmin(delta_v_grid)
    vmax = np.nanmin(delta_v_grid) + 10  # Show up to 10 km/s above minimum
    levels = np.linspace(vmin, vmax, 50)
    
    # Create date format for x-axis
    date_format = '%Y-%m-%d'
    dep_dates = [d.strftime(date_format) for d in departure_dates]
    arr_dates = [arrival_dates[0, j].strftime(date_format) for j in range(arrival_dates.shape[1])]
    
    # Convert dates to numerical format for plotting
    dep_nums = np.arange(len(dep_dates))
    arr_nums = np.arange(len(arr_dates))
    
    # Create mesh for contour plot
    X, Y = np.meshgrid(dep_nums, arr_nums)
    
    # Plot delta-v contours
    cs1 = ax1.contourf(X.T, Y.T, delta_v_grid, levels=levels, cmap='viridis')
    fig.colorbar(cs1, ax=ax1, label='Total ΔV (km/s)')
    
    # Mark minimum point
    ax1.scatter(min_idx[0], min_idx[1], c='red', marker='*', s=200, 
                edgecolor='white', label=f'Min ΔV: {min_dv:.2f} km/s')
    
    # Set labels and title for first plot
    ax1.set_title('Earth-Mars Transfer: Total ΔV')
    ax1.set_xlabel('Earth Departure Date')
    ax1.set_ylabel('Mars Arrival Date')
    
    # Set x-ticks
    tick_step = max(1, len(dep_dates) // 10)
    ax1.set_xticks(dep_nums[::tick_step])
    ax1.set_xticklabels(dep_dates[::tick_step], rotation=45)
    
    # Set y-ticks
    tick_step = max(1, len(arr_dates) // 10)
    ax1.set_yticks(arr_nums[::tick_step])
    ax1.set_yticklabels(arr_dates[::tick_step])
    
    # Add time of flight contours
    tof_days = np.array([(arrival_dates[i, j] - departure_dates[i]).days 
                         for i in range(len(departure_dates))
                         for j in range(arrival_dates.shape[1])]).reshape(delta_v_grid.shape)
    
    tof_levels = np.arange(min(100, np.nanmin(tof_days)), np.nanmax(tof_days), 50)
    cs3 = ax1.contour(X.T, Y.T, tof_days, levels=tof_levels, colors='white', alpha=0.5)
    ax1.clabel(cs3, inline=True, fontsize=8, fmt='%d days')
    
    # Create legend
    ax1.legend(loc='upper right')
    
    # Plot C3 contours
    vmin_c3 = np.nanmin(c3_grid)
    vmax_c3 = vmin_c3 + 100  # Show up to 100 km²/s² above minimum
    levels_c3 = np.linspace(vmin_c3, vmax_c3, 50)
    
    cs2 = ax2.contourf(X.T, Y.T, c3_grid, levels=levels_c3, cmap='plasma')
    fig.colorbar(cs2, ax=ax2, label='C3 Energy (km²/s²)')
    
    # Mark minimum point
    min_c3 = np.nanmin(c3_grid)
    min_idx_c3 = np.unravel_index(np.nanargmin(c3_grid), c3_grid.shape)
    ax2.scatter(min_idx_c3[0], min_idx_c3[1], c='red', marker='*', s=200, 
                edgecolor='white', label=f'Min C3: {min_c3:.2f} km²/s²')
    
    # Set labels and title for second plot
    ax2.set_title('Earth-Mars Transfer: C3 Energy')
    ax2.set_xlabel('Earth Departure Date')
    ax2.set_ylabel('Mars Arrival Date')
    
    # Set x-ticks
    ax2.set_xticks(dep_nums[::tick_step])
    ax2.set_xticklabels(dep_dates[::tick_step], rotation=45)
    
    # Set y-ticks
    ax2.set_yticks(arr_nums[::tick_step])
    ax2.set_yticklabels(arr_dates[::tick_step])
    
    # Add time of flight contours
    cs4 = ax2.contour(X.T, Y.T, tof_days, levels=tof_levels, colors='white', alpha=0.5)
    ax2.clabel(cs4, inline=True, fontsize=8, fmt='%d days')
    
    # Create legend
    ax2.legend(loc='upper right')
    
    # Print optimal parameters
    print("\nOptimal Transfer Window:")
    print(f"Departure Date: {min_departure.strftime(date_format)}")
    print(f"Arrival Date: {min_arrival.strftime(date_format)}")
    print(f"Time of Flight: {min_tof} days")
    print(f"Total ΔV: {min_dv:.2f} km/s")
    
    # Add overall title with optimal parameters
    plt.suptitle(f'Earth-Mars Porkchop Plot\nOptimal: Depart {min_departure.strftime(date_format)}, '
                 f'Arrive {min_arrival.strftime(date_format)}, '
                 f'TOF: {min_tof} days, ΔV: {min_dv:.2f} km/s',
                 fontsize=16)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Porkchop plot saved to {filename}")
    return fig

def main():
    """Main function to generate porkchop plot."""
    print("Mars Porkchop Plot Generator")
    print("===========================")
    
    try:
        # Define date ranges
        # Use a shorter window around an expected good launch opportunity
        start_date = dt.datetime(2022, 8, 1)  # August 2022
        end_date = dt.datetime(2022, 12, 1)   # December 2022
        min_tof = 180  # Minimum time of flight (days)
        max_tof = 280  # Maximum time of flight (days)
        
        print(f"Calculating transfers for departures from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Time of flight range: {min_tof} to {max_tof} days")
        
        # Using larger step sizes for faster computation
        step_days = 5        # 5 days between departure dates (reduced from 20)
        tof_step_days = 5    # 5 days between TOF values (reduced from 20)
        
        print(f"Grid resolution: {step_days} days for departures, {tof_step_days} days for TOF")
        print(f"Computing approximately {(end_date-start_date).days//step_days * (max_tof-min_tof)//tof_step_days} points")
        
        # Calculate porkchop data
        departure_dates, arrival_dates, delta_v_grid, c3_grid = calculate_porkchop_data(
            start_date, end_date, min_tof, max_tof, step_days=step_days, tof_step_days=tof_step_days)
        
        # Plot porkchop
        plot_porkchop(departure_dates, arrival_dates, delta_v_grid, c3_grid)
        
        print("\nPorkchop plot generation completed successfully!")
        
    except Exception as e:
        print(f"Error generating porkchop plot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 