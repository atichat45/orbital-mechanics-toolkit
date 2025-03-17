#!/usr/bin/env python3
"""
Numerical Integration and Orbit Propagation Script

This script demonstrates numerical integration of orbital equations of motion
to propagate orbits under various force models, including:
- Two-body problem
- J2 perturbation
- Solar radiation pressure
- Third-body gravity
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.integrate import solve_ivp
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.orbital_mechanics.core.orbital_elements import (
    rv_to_orbital_elements,
    orbital_elements_to_rv,
    true_anomaly_from_eccentric_anomaly,
    eccentric_anomaly_from_mean_anomaly
)
from src.orbital_mechanics.data.spice_interface import SpiceInterface
from src.orbital_mechanics.utils.constants import PLANETS, AU, EARTH_RADIUS, DAY
from src.orbital_mechanics.visualization.orbit_plotting import plot_trajectory_3d

def run_numerical_integration_analysis():
    """
    Run a comprehensive analysis of numerical integration methods
    for orbit propagation.
    """
    print("Numerical Integration and Orbit Propagation Analysis")
    print("==================================================")
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize SPICE interface for reference orbits
    print("Initializing SPICE interface...")
    spice = SpiceInterface()
    spice.load_kernels()
    
    try:
        # Set up scenarios to analyze
        analyze_leo_orbit_with_perturbations(output_dir)
        analyze_interplanetary_trajectory(spice, output_dir)
        
    finally:
        # Unload SPICE kernels
        spice.unload_kernels()
        print("\nAnalysis completed successfully!")

def analyze_leo_orbit_with_perturbations(output_dir):
    """
    Analyze a LEO orbit with various perturbation models.
    
    Args:
        output_dir: Directory to save outputs
    """
    print("\n1. Low Earth Orbit with Perturbations")
    print("------------------------------------")
    
    # Initial orbital elements for a typical LEO satellite
    a = EARTH_RADIUS + 500e3  # 500 km altitude
    e = 0.0001  # near-circular
    i = np.radians(51.6)  # ISS-like inclination
    omega = np.radians(0)  # argument of periapsis
    Omega = np.radians(0)  # right ascension of ascending node
    nu = np.radians(0)  # true anomaly
    
    # Convert to Cartesian state vector
    r0, v0 = orbital_elements_to_rv(a, e, i, omega, Omega, nu, PLANETS['Earth']['mu'])
    
    # Initial state vector [x, y, z, vx, vy, vz]
    y0 = np.concatenate([r0, v0])
    
    # Time span: 1 day with 1-minute steps
    t_span = (0, DAY)
    t_eval = np.linspace(0, DAY, 1440)  # 1-minute steps
    
    # Run simulations with different force models
    results = {}
    
    print("Running numerical integration with various force models...")
    
    # Two-body problem (no perturbations)
    sol_two_body = solve_ivp(
        lambda t, y: two_body_dynamics(t, y, PLANETS['Earth']['mu']),
        t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-10
    )
    results['Two-Body'] = sol_two_body
    
    # With J2 perturbation
    sol_j2 = solve_ivp(
        lambda t, y: j2_perturbed_dynamics(t, y, PLANETS['Earth']['mu'], EARTH_RADIUS, PLANETS['Earth']['J2']),
        t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-10
    )
    results['J2-Perturbed'] = sol_j2
    
    # With J2 and drag (simplified model)
    sol_j2_drag = solve_ivp(
        lambda t, y: j2_drag_perturbed_dynamics(
            t, y, PLANETS['Earth']['mu'], EARTH_RADIUS, PLANETS['Earth']['J2'],
            rho0=1.0e-11, scale_height=8500, drag_coefficient=2.2, mass=100, area=1
        ),
        t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-10
    )
    results['J2+Drag'] = sol_j2_drag
    
    # With multiple perturbations (J2, drag, SRP, third-body)
    sol_full = solve_ivp(
        lambda t, y: full_perturbed_dynamics(
            t, y, PLANETS['Earth']['mu'], EARTH_RADIUS, PLANETS['Earth']['J2'],
            rho0=1.0e-11, scale_height=8500, drag_coefficient=2.2, mass=100, area=1,
            srp_coefficient=1.8, solar_flux=1361, speed_of_light=299792458,
            third_body_mu=PLANETS['Moon']['mu'], r_third=np.array([384400e3, 0, 0])
        ),
        t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-10
    )
    results['Full-Perturbation'] = sol_full
    
    # Plot results
    print("Generating visualizations...")
    
    # 3D trajectory plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw Earth (not to scale, for visualization)
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = EARTH_RADIUS * np.cos(u) * np.sin(v)
    y = EARTH_RADIUS * np.sin(u) * np.sin(v)
    z = EARTH_RADIUS * np.cos(v)
    ax.plot_surface(x, y, z, color='blue', alpha=0.2)
    
    # Plot each trajectory
    colors = ['darkblue', 'green', 'red', 'purple']
    for (name, sol), color in zip(results.items(), colors):
        ax.plot(sol.y[0], sol.y[1], sol.y[2], label=name, color=color, linewidth=2)
    
    # Configure plot
    max_limit = 1.2 * a
    ax.set_xlim(-max_limit, max_limit)
    ax.set_ylim(-max_limit, max_limit)
    ax.set_zlim(-max_limit, max_limit)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    ax.set_title('LEO Orbit Propagation with Various Perturbations')
    ax.legend(loc='upper right')
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'leo_perturbations_3d.png'), dpi=300, bbox_inches='tight')
    print("3D visualization saved as 'leo_perturbations_3d.png'")
    
    # Time series analysis of orbital elements
    fig, axs = plt.subplots(3, 2, figsize=(16, 12))
    
    # Calculate orbital elements over time for each model
    for (name, sol), color in zip(results.items(), colors):
        times = sol.t / 3600  # convert to hours for plotting
        
        # Initialize arrays to store orbital elements
        a_vals = np.zeros(len(times))
        e_vals = np.zeros(len(times))
        i_vals = np.zeros(len(times))
        omega_vals = np.zeros(len(times))
        Omega_vals = np.zeros(len(times))
        nu_vals = np.zeros(len(times))
        
        # Calculate orbital elements at each time step
        for j in range(len(times)):
            r = sol.y[:3, j]
            v = sol.y[3:, j]
            
            # Convert to orbital elements
            elements = rv_to_orbital_elements(r, v, PLANETS['Earth']['mu'])
            a_vals[j] = elements[0]
            e_vals[j] = elements[1]
            i_vals[j] = np.degrees(elements[2])  # convert to degrees
            omega_vals[j] = np.degrees(elements[3])
            Omega_vals[j] = np.degrees(elements[4])
            nu_vals[j] = np.degrees(elements[5])
        
        # Plot semi-major axis
        axs[0, 0].plot(times, (a_vals - a) * 1000, label=name, color=color)  # in meters
        axs[0, 0].set_title('Semi-major Axis Change')
        axs[0, 0].set_xlabel('Time (hours)')
        axs[0, 0].set_ylabel('Δa (m)')
        axs[0, 0].grid(True, alpha=0.3)
        
        # Plot eccentricity
        axs[0, 1].plot(times, e_vals, label=name, color=color)
        axs[0, 1].set_title('Eccentricity')
        axs[0, 1].set_xlabel('Time (hours)')
        axs[0, 1].set_ylabel('e')
        axs[0, 1].grid(True, alpha=0.3)
        
        # Plot inclination
        axs[1, 0].plot(times, i_vals, label=name, color=color)
        axs[1, 0].set_title('Inclination')
        axs[1, 0].set_xlabel('Time (hours)')
        axs[1, 0].set_ylabel('i (deg)')
        axs[1, 0].grid(True, alpha=0.3)
        
        # Plot argument of periapsis
        axs[1, 1].plot(times, omega_vals, label=name, color=color)
        axs[1, 1].set_title('Argument of Periapsis')
        axs[1, 1].set_xlabel('Time (hours)')
        axs[1, 1].set_ylabel('ω (deg)')
        axs[1, 1].grid(True, alpha=0.3)
        
        # Plot RAAN
        axs[2, 0].plot(times, Omega_vals, label=name, color=color)
        axs[2, 0].set_title('Right Ascension of Ascending Node')
        axs[2, 0].set_xlabel('Time (hours)')
        axs[2, 0].set_ylabel('Ω (deg)')
        axs[2, 0].grid(True, alpha=0.3)
        
        # Plot true anomaly
        axs[2, 1].plot(times, nu_vals, label=name, color=color)
        axs[2, 1].set_title('True Anomaly')
        axs[2, 1].set_xlabel('Time (hours)')
        axs[2, 1].set_ylabel('ν (deg)')
        axs[2, 1].grid(True, alpha=0.3)
    
    # Add legend to the first subplot only
    axs[0, 0].legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'leo_orbital_elements.png'), dpi=300, bbox_inches='tight')
    print("Orbital elements analysis saved as 'leo_orbital_elements.png'")
    
    # Export data for further analysis
    data = {
        'Time (hours)': sol_two_body.t / 3600
    }
    
    for name, sol in results.items():
        data[f'{name}_X (m)'] = sol.y[0]
        data[f'{name}_Y (m)'] = sol.y[1]
        data[f'{name}_Z (m)'] = sol.y[2]
        data[f'{name}_VX (m/s)'] = sol.y[3]
        data[f'{name}_VY (m/s)'] = sol.y[4]
        data[f'{name}_VZ (m/s)'] = sol.y[5]
    
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, 'leo_propagation_results.csv'), index=False)
    print("Propagation data saved to 'leo_propagation_results.csv'")

def analyze_interplanetary_trajectory(spice, output_dir):
    """
    Analyze an interplanetary trajectory using numerical integration and
    compare with SPICE ephemeris.
    
    Args:
        spice: Initialized SpiceInterface
        output_dir: Directory to save outputs
    """
    print("\n2. Interplanetary Trajectory Analysis")
    print("---------------------------------")
    
    # Define start and end dates for an Earth-Mars transfer
    start_date = datetime(2022, 1, 1)
    end_date = start_date + timedelta(days=240)  # Approximate transfer time
    
    # Get Earth state vector at departure (relative to Sun)
    r_earth, v_earth = spice.get_state(
        target='EARTH',
        observer='SUN',
        datetime_obj=start_date
    )
    
    # Get Mars state vector at arrival (relative to Sun)
    r_mars, v_mars = spice.get_state(
        target='MARS',
        observer='SUN',
        datetime_obj=end_date
    )
    
    print(f"Analyzing Earth-Mars transfer from {start_date.strftime('%Y-%m-%d')} "
          f"to {end_date.strftime('%Y-%m-%d')}")
    
    # Compute Lambert arc
    from src.orbital_mechanics.core.lambert import solve_lambert
    
    tof = (end_date - start_date).total_seconds()
    v1, v2 = solve_lambert(
        r1=r_earth,
        r2=r_mars,
        tof=tof,
        mu=PLANETS['Sun']['mu'],
        clockwise=False
    )
    
    print(f"Lambert solution computed:")
    print(f"  Departure velocity: {np.linalg.norm(v1)/1000:.2f} km/s")
    print(f"  Arrival velocity: {np.linalg.norm(v2)/1000:.2f} km/s")
    
    # Calculate delta-V
    delta_v_departure = np.linalg.norm(v1 - v_earth)
    delta_v_arrival = np.linalg.norm(v_mars - v2)
    
    print(f"  Departure ΔV: {delta_v_departure/1000:.2f} km/s")
    print(f"  Arrival ΔV: {delta_v_arrival/1000:.2f} km/s")
    print(f"  Total ΔV: {(delta_v_departure + delta_v_arrival)/1000:.2f} km/s")
    
    # Initial state for propagation
    y0 = np.concatenate([r_earth, v1])
    
    # Time span
    t_span = (0, tof)
    
    # Create evaluation times - one point per day
    num_days = int(tof / DAY)
    t_eval = np.linspace(0, tof, num_days + 1)
    
    # Run numerical integration
    print("Running numerical integration of the transfer trajectory...")
    
    # Two-body problem (Sun only)
    sol_two_body = solve_ivp(
        lambda t, y: two_body_dynamics(t, y, PLANETS['Sun']['mu']),
        t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-10
    )
    
    # Run with n-body problem (Sun, Jupiter, Venus)
    print("Running n-body simulation including gravitational effects of major planets...")
    
    def n_body_dynamics_wrapper(t, y):
        """Wrapper to include time-dependent planet positions"""
        current_time = start_date + timedelta(seconds=t)
        
        # Get positions of major planets
        jupiter_pos = spice.get_position('JUPITER BARYCENTER', 'SUN', current_time)
        venus_pos = spice.get_position('VENUS BARYCENTER', 'SUN', current_time)
        
        return n_body_dynamics(
            t, y, 
            PLANETS['Sun']['mu'],
            [(PLANETS['Jupiter']['mu'], jupiter_pos),
             (PLANETS['Venus']['mu'], venus_pos)]
        )
    
    sol_n_body = solve_ivp(
        n_body_dynamics_wrapper,
        t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-10
    )
    
    # Get arrival state
    r_final_two_body = sol_two_body.y[:3, -1]
    v_final_two_body = sol_two_body.y[3:, -1]
    
    r_final_n_body = sol_n_body.y[:3, -1]
    v_final_n_body = sol_n_body.y[3:, -1]
    
    # Calculate miss distances
    miss_two_body = np.linalg.norm(r_final_two_body - r_mars)
    miss_n_body = np.linalg.norm(r_final_n_body - r_mars)
    
    print(f"Two-body propagation miss distance: {miss_two_body/1000:.2f} km")
    print(f"N-body propagation miss distance: {miss_n_body/1000:.2f} km")
    
    # Generate comparison points from SPICE for validation
    spice_positions = []
    
    for i, t in enumerate(t_eval):
        current_time = start_date + timedelta(seconds=t)
        
        # Get Earth and Mars positions at this time
        earth_pos = spice.get_position('EARTH', 'SUN', current_time)
        mars_pos = spice.get_position('MARS', 'SUN', current_time)
        
        spice_positions.append((earth_pos, mars_pos))
    
    spice_positions = np.array(spice_positions)
    
    # Plot results
    print("Generating trajectory visualizations...")
    
    # Plot the 3D trajectories
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Sun
    ax.scatter([0], [0], [0], color='gold', s=200, label='Sun')
    
    # Plot Earth orbit
    earth_orbit = []
    for i in range(365):
        t = start_date + timedelta(days=i)
        pos = spice.get_position('EARTH', 'SUN', t)
        earth_orbit.append(pos)
    
    earth_orbit = np.array(earth_orbit)
    ax.plot(earth_orbit[:, 0]/AU, earth_orbit[:, 1]/AU, earth_orbit[:, 2]/AU, 
            'b-', alpha=0.5, label="Earth's Orbit")
    
    # Plot Mars orbit
    mars_orbit = []
    for i in range(687):  # Mars orbital period
        t = start_date + timedelta(days=i)
        pos = spice.get_position('MARS', 'SUN', t)
        mars_orbit.append(pos)
    
    mars_orbit = np.array(mars_orbit)
    ax.plot(mars_orbit[:, 0]/AU, mars_orbit[:, 1]/AU, mars_orbit[:, 2]/AU, 
            'r-', alpha=0.5, label="Mars' Orbit")
    
    # Plot Earth at departure
    ax.scatter([r_earth[0]/AU], [r_earth[1]/AU], [r_earth[2]/AU], 
               color='blue', s=100, label='Earth at Departure')
    
    # Plot Mars at arrival
    ax.scatter([r_mars[0]/AU], [r_mars[1]/AU], [r_mars[2]/AU], 
               color='red', s=100, label='Mars at Arrival')
    
    # Plot two-body trajectory
    ax.plot(sol_two_body.y[0]/AU, sol_two_body.y[1]/AU, sol_two_body.y[2]/AU, 
            'g-', linewidth=2, label='Two-Body Trajectory')
    
    # Plot n-body trajectory
    ax.plot(sol_n_body.y[0]/AU, sol_n_body.y[1]/AU, sol_n_body.y[2]/AU, 
            'm--', linewidth=2, label='N-Body Trajectory')
    
    # Configure plot
    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    
    ax.set_title(f'Earth-Mars Transfer Trajectory\n'
                f'Departure: {start_date.strftime("%Y-%m-%d")}, '
                f'Arrival: {end_date.strftime("%Y-%m-%d")}, '
                f'Duration: {num_days} days')
    
    # Improve viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Add a legend
    ax.legend(loc='upper right')
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'interplanetary_trajectory.png'), dpi=300, bbox_inches='tight')
    print("Interplanetary trajectory visualization saved as 'interplanetary_trajectory.png'")
    
    # Plot error analysis
    fig, axs = plt.subplots(2, 1, figsize=(14, 10))
    
    # Time in days for plotting
    days = t_eval / DAY
    
    # Calculate positional errors compared to Lambert arc
    lambert_positions = np.zeros((len(t_eval), 3))
    
    for i, t in enumerate(t_eval):
        # Propagate Lambert arc analytically
        fraction = t / tof
        lambert_positions[i] = r_earth * (1 - fraction) + r_mars * fraction  # Crude approximation
    
    # Calculate positional errors
    error_two_body = np.zeros(len(t_eval))
    error_n_body = np.zeros(len(t_eval))
    
    for i in range(len(t_eval)):
        error_two_body[i] = np.linalg.norm(sol_two_body.y[:3, i] - lambert_positions[i]) / 1000  # km
        error_n_body[i] = np.linalg.norm(sol_n_body.y[:3, i] - lambert_positions[i]) / 1000  # km
    
    # Plot position error
    axs[0].plot(days, error_two_body, 'g-', label='Two-Body Error', linewidth=2)
    axs[0].plot(days, error_n_body, 'm--', label='N-Body Error', linewidth=2)
    axs[0].set_title('Position Error Relative to Lambert Arc')
    axs[0].set_xlabel('Time (days)')
    axs[0].set_ylabel('Error (km)')
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    
    # Calculate distance from Earth and Mars
    dist_to_earth = np.zeros((len(t_eval), 2))  # [two_body, n_body]
    dist_to_mars = np.zeros((len(t_eval), 2))  # [two_body, n_body]
    
    for i in range(len(t_eval)):
        earth_pos = spice_positions[i, 0]
        mars_pos = spice_positions[i, 1]
        
        dist_to_earth[i, 0] = np.linalg.norm(sol_two_body.y[:3, i] - earth_pos) / AU  # AU
        dist_to_earth[i, 1] = np.linalg.norm(sol_n_body.y[:3, i] - earth_pos) / AU  # AU
        
        dist_to_mars[i, 0] = np.linalg.norm(sol_two_body.y[:3, i] - mars_pos) / AU  # AU
        dist_to_mars[i, 1] = np.linalg.norm(sol_n_body.y[:3, i] - mars_pos) / AU  # AU
    
    # Plot distances
    axs[1].plot(days, dist_to_earth[:, 0], 'b-', label='Distance to Earth (Two-Body)', linewidth=2)
    axs[1].plot(days, dist_to_earth[:, 1], 'b--', label='Distance to Earth (N-Body)', linewidth=2)
    axs[1].plot(days, dist_to_mars[:, 0], 'r-', label='Distance to Mars (Two-Body)', linewidth=2)
    axs[1].plot(days, dist_to_mars[:, 1], 'r--', label='Distance to Mars (N-Body)', linewidth=2)
    
    axs[1].set_title('Distance to Earth and Mars')
    axs[1].set_xlabel('Time (days)')
    axs[1].set_ylabel('Distance (AU)')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'interplanetary_error_analysis.png'), dpi=300, bbox_inches='tight')
    print("Error analysis saved as 'interplanetary_error_analysis.png'")
    
    # Export data
    interplanetary_data = {
        'Time (days)': days,
        'Two_Body_X (AU)': sol_two_body.y[0] / AU,
        'Two_Body_Y (AU)': sol_two_body.y[1] / AU,
        'Two_Body_Z (AU)': sol_two_body.y[2] / AU,
        'N_Body_X (AU)': sol_n_body.y[0] / AU,
        'N_Body_Y (AU)': sol_n_body.y[1] / AU,
        'N_Body_Z (AU)': sol_n_body.y[2] / AU,
        'Error_Two_Body (km)': error_two_body,
        'Error_N_Body (km)': error_n_body,
        'Earth_Distance_Two_Body (AU)': dist_to_earth[:, 0],
        'Earth_Distance_N_Body (AU)': dist_to_earth[:, 1],
        'Mars_Distance_Two_Body (AU)': dist_to_mars[:, 0],
        'Mars_Distance_N_Body (AU)': dist_to_mars[:, 1],
    }
    
    interplanetary_df = pd.DataFrame(interplanetary_data)
    interplanetary_df.to_csv(os.path.join(output_dir, 'interplanetary_propagation_results.csv'), index=False)
    print("Interplanetary propagation data saved to 'interplanetary_propagation_results.csv'")

# Force model functions for numerical integration

def two_body_dynamics(t, y, mu):
    """
    Two-body problem dynamics.
    
    Args:
        t: Time (not used in time-invariant systems)
        y: State vector [x, y, z, vx, vy, vz]
        mu: Gravitational parameter
        
    Returns:
        dy/dt: Derivative of state vector
    """
    r = y[:3]
    v = y[3:]
    
    # Distance
    r_norm = np.linalg.norm(r)
    
    # Acceleration from gravity
    a = -mu * r / r_norm**3
    
    return np.concatenate([v, a])

def j2_perturbed_dynamics(t, y, mu, radius, j2):
    """
    Dynamics with J2 perturbation.
    
    Args:
        t: Time (not used in time-invariant systems)
        y: State vector [x, y, z, vx, vy, vz]
        mu: Gravitational parameter
        radius: Central body radius
        j2: J2 coefficient
        
    Returns:
        dy/dt: Derivative of state vector
    """
    r = y[:3]
    v = y[3:]
    
    # Distance
    r_norm = np.linalg.norm(r)
    
    # Two-body acceleration
    a_two_body = -mu * r / r_norm**3
    
    # J2 perturbation
    x, y, z = r
    r2 = r_norm**2
    z2 = z**2
    
    factor = 3 * j2 * mu * radius**2 / (2 * r_norm**5)
    
    ax_j2 = factor * x * (5 * z2 / r2 - 1)
    ay_j2 = factor * y * (5 * z2 / r2 - 1)
    az_j2 = factor * z * (5 * z2 / r2 - 3)
    
    a_j2 = np.array([ax_j2, ay_j2, az_j2])
    
    # Total acceleration
    a = a_two_body + a_j2
    
    return np.concatenate([v, a])

def j2_drag_perturbed_dynamics(t, y, mu, radius, j2, rho0, scale_height, drag_coefficient, mass, area):
    """
    Dynamics with J2 and atmospheric drag perturbations.
    
    Args:
        t: Time (not used in time-invariant systems)
        y: State vector [x, y, z, vx, vy, vz]
        mu: Gravitational parameter
        radius: Central body radius
        j2: J2 coefficient
        rho0: Atmospheric density at reference altitude (kg/m^3)
        scale_height: Scale height for exponential atmosphere model (m)
        drag_coefficient: Drag coefficient
        mass: Spacecraft mass (kg)
        area: Spacecraft cross-sectional area (m^2)
        
    Returns:
        dy/dt: Derivative of state vector
    """
    r = y[:3]
    v = y[3:]
    
    # Distance
    r_norm = np.linalg.norm(r)
    
    # Two-body and J2 acceleration
    a_j2 = j2_perturbed_dynamics(t, y, mu, radius, j2)[3:]
    
    # Drag model
    v_norm = np.linalg.norm(v)
    
    # Altitude above surface
    altitude = r_norm - radius
    
    # Atmospheric density (simple exponential model)
    rho = rho0 * np.exp(-altitude / scale_height)
    
    # Drag acceleration magnitude
    drag_factor = -0.5 * drag_coefficient * area * rho * v_norm / mass
    
    # Drag acceleration vector (opposite to velocity)
    a_drag = drag_factor * v
    
    # Total acceleration
    a = a_j2 + a_drag
    
    return np.concatenate([v, a])

def full_perturbed_dynamics(t, y, mu, radius, j2, rho0, scale_height, drag_coefficient, mass, area,
                            srp_coefficient, solar_flux, speed_of_light, third_body_mu, r_third):
    """
    Dynamics with multiple perturbations: J2, drag, SRP, third-body gravity.
    
    Args:
        t: Time (not used in time-invariant systems)
        y: State vector [x, y, z, vx, vy, vz]
        mu: Gravitational parameter
        radius: Central body radius
        j2: J2 coefficient
        rho0: Atmospheric density at reference altitude (kg/m^3)
        scale_height: Scale height for exponential atmosphere model (m)
        drag_coefficient: Drag coefficient
        mass: Spacecraft mass (kg)
        area: Spacecraft cross-sectional area (m^2)
        srp_coefficient: Solar radiation pressure coefficient
        solar_flux: Solar flux at spacecraft location (W/m^2)
        speed_of_light: Speed of light (m/s)
        third_body_mu: Gravitational parameter of third body
        r_third: Position vector of third body [x, y, z]
        
    Returns:
        dy/dt: Derivative of state vector
    """
    r = y[:3]
    v = y[3:]
    
    # Get acceleration due to J2 and drag
    a_j2_drag = j2_drag_perturbed_dynamics(
        t, y, mu, radius, j2, rho0, scale_height, drag_coefficient, mass, area
    )[3:]
    
    # Solar radiation pressure
    # Direction from Sun to spacecraft (assuming Sun at origin for simplicity)
    r_sun = np.array([0, 0, 0]) - r  # Vector from spacecraft to Sun
    r_sun_norm = np.linalg.norm(r_sun)
    r_sun_unit = r_sun / r_sun_norm
    
    # SRP acceleration (in direction away from Sun)
    srp_magnitude = srp_coefficient * solar_flux * area / (mass * speed_of_light)
    a_srp = -srp_magnitude * r_sun_unit  # Negative because force is away from Sun
    
    # Third-body gravitational perturbation (e.g., Moon)
    # Vector from spacecraft to third body
    r_sc_to_third = r_third - r
    r_sc_to_third_norm = np.linalg.norm(r_sc_to_third)
    
    # Third-body acceleration
    a_third_body = third_body_mu * (
        r_sc_to_third / r_sc_to_third_norm**3 - 
        r_third / np.linalg.norm(r_third)**3
    )
    
    # Total acceleration
    a = a_j2_drag + a_srp + a_third_body
    
    return np.concatenate([v, a])

def n_body_dynamics(t, y, central_mu, bodies):
    """
    N-body problem dynamics.
    
    Args:
        t: Time
        y: State vector [x, y, z, vx, vy, vz]
        central_mu: Gravitational parameter of central body
        bodies: List of tuples (mu, r) for perturbing bodies
        
    Returns:
        dy/dt: Derivative of state vector
    """
    r = y[:3]
    v = y[3:]
    
    # Primary two-body acceleration (central body)
    r_norm = np.linalg.norm(r)
    a = -central_mu * r / r_norm**3
    
    # Add accelerations from other bodies
    for mu_body, r_body in bodies:
        # Vector from spacecraft to body
        r_sc_to_body = r_body - r
        r_sc_to_body_norm = np.linalg.norm(r_sc_to_body)
        
        # Acceleration due to this body (using the indirect term for proper formulation)
        a += mu_body * (
            r_sc_to_body / r_sc_to_body_norm**3 - 
            r_body / np.linalg.norm(r_body)**3
        )
    
    return np.concatenate([v, a])

if __name__ == "__main__":
    run_numerical_integration_analysis() 