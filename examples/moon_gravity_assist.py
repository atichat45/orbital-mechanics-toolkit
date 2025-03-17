"""
Lunar Gravity Assist Example

This script demonstrates how to use the orbital mechanics toolkit to:
1. Plan a mission using the current date as the starting point
2. Implement a lunar gravity assist to accelerate a spacecraft to Mars
3. Visualize and analyze the complete trajectory
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

from orbital_mechanics.core.lambert import solve_lambert
from orbital_mechanics.visualization.orbit_plotting import (
    setup_solar_system_plot, 
    plot_planet_orbit, 
    plot_trajectory_3d
)
from orbital_mechanics.utils.constants import (
    AU, DAY, YEAR, SUN_MU, EARTH_MU, MOON_MU, 
    PLANETS, EARTH_RADIUS, MOON_RADIUS, MOON_SEMI_MAJOR_AXIS
)

# Output directory for saving results
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def two_body_equation(t, y, mu):
    """
    Two-body equation of motion for numerical integration.
    
    Args:
        t: Time (not used, required by solve_ivp)
        y: State vector [x, y, z, vx, vy, vz]
        mu: Gravitational parameter
    
    Returns:
        Derivatives [vx, vy, vz, ax, ay, az]
    """
    r = y[0:3]
    v = y[3:6]
    
    r_norm = np.linalg.norm(r)
    
    # Check for division by zero
    if r_norm < 1e-10:
        r_norm = 1e-10
    
    # Acceleration components
    a = -mu * r / r_norm**3
    
    return np.concatenate((v, a))

def propagate_orbit(r0, v0, tof, mu, steps=100):
    """
    Numerically propagate an orbit from initial state for specified time.
    
    Args:
        r0: Initial position vector [x, y, z] (m)
        v0: Initial velocity vector [vx, vy, vz] (m/s)
        tof: Time of flight (s)
        mu: Gravitational parameter (m^3/s^2)
        steps: Number of steps for output
    
    Returns:
        Dictionary with trajectory data
    """
    # Initial state vector
    y0 = np.concatenate((r0, v0))
    
    # Time span
    t_span = (0, tof)
    t_eval = np.linspace(0, tof, steps)
    
    # Numerical integration
    try:
        solution = solve_ivp(
            two_body_equation, 
            t_span, 
            y0, 
            args=(mu,), 
            method='RK45', 
            t_eval=t_eval,
            rtol=1e-8, 
            atol=1e-8
        )
        
        trajectory = solution.y.T
        times = solution.t
        
        return {
            'trajectory': trajectory,
            'times': times,
            'success': solution.success
        }
    except Exception as e:
        print(f"Error in orbit propagation: {e}")
        return None

def main():
    print("Note: This is a simplified demonstration using ideal circular orbits.")
    print("Real mission planning would use precise ephemeris data from SPICE.")
    
    # Use current date as mission start date
    mission_start = datetime.now()
    print(f"\nPlanning mission starting from current date: {mission_start.strftime('%Y-%m-%d')}")
    
    # Step 1: Setup Earth-Moon system
    print("\nStep 1: Setting up Earth-Moon system")
    earth_moon_dist = MOON_SEMI_MAJOR_AXIS  # m
    
    # Earth's position at the Sun (assuming circular orbit for simplicity)
    days_since_j2000 = (mission_start - datetime(2000, 1, 1)).days
    earth_angle = (days_since_j2000 / 365.25) * 2 * np.pi  # Earth's position around the Sun
    earth_pos = np.array([
        PLANETS['Earth']['semi_major_axis'] * np.cos(earth_angle),
        PLANETS['Earth']['semi_major_axis'] * np.sin(earth_angle),
        0.0
    ])
    
    # Earth's velocity (circular orbit approximation)
    earth_speed = np.sqrt(SUN_MU / PLANETS['Earth']['semi_major_axis'])
    earth_vel = np.array([
        -earth_speed * np.sin(earth_angle),
        earth_speed * np.cos(earth_angle),
        0.0
    ])
    
    # Moon's position relative to Earth (simplified)
    # In a real application, this would come from ephemeris data
    # Position the Moon in a favorable position for the transfer
    moon_angle = earth_angle + np.pi/4  # 45 degrees ahead of Earth
    moon_pos_rel = np.array([
        earth_moon_dist * np.cos(moon_angle),
        earth_moon_dist * np.sin(moon_angle),
        0.0
    ])
    
    # Moon's position in the solar system
    moon_pos = earth_pos + moon_pos_rel
    
    # Moon's velocity around Earth
    moon_speed_around_earth = np.sqrt(EARTH_MU / earth_moon_dist)
    moon_vel_rel = np.array([
        -moon_speed_around_earth * np.sin(moon_angle),
        moon_speed_around_earth * np.cos(moon_angle),
        0.0
    ])
    
    # Moon's velocity in the solar system (Earth's velocity + Moon's relative velocity)
    moon_vel = earth_vel + moon_vel_rel
    
    print(f"Earth position (AU): [{earth_pos[0]/AU:.4f}, {earth_pos[1]/AU:.4f}, {earth_pos[2]/AU:.4f}]")
    print(f"Moon position (AU): [{moon_pos[0]/AU:.4f}, {moon_pos[1]/AU:.4f}, {moon_pos[2]/AU:.4f}]")
    
    # Step 2: Calculate Mars position at a future arrival date
    # For demonstration purposes, place Mars at a reasonable distance 
    # and angle for a simplified transfer calculation
    mars_arrival_date = mission_start + timedelta(days=260)  # Typical Earth-Mars transfer time
    
    # Place Mars at a favorable position for the transfer
    mars_angle = earth_angle + np.pi * 0.8  # ~145 degrees ahead, good for Hohmann-like transfer
    mars_pos = np.array([
        PLANETS['Mars']['semi_major_axis'] * np.cos(mars_angle),
        PLANETS['Mars']['semi_major_axis'] * np.sin(mars_angle),
        0.0
    ])
    
    mars_speed = np.sqrt(SUN_MU / PLANETS['Mars']['semi_major_axis'])
    mars_vel = np.array([
        -mars_speed * np.sin(mars_angle),
        mars_speed * np.cos(mars_angle),
        0.0
    ])
    
    print(f"\nStep 2: Calculating Mars position at arrival date: {mars_arrival_date.strftime('%Y-%m-%d')}")
    print(f"Mars position (AU): [{mars_pos[0]/AU:.4f}, {mars_pos[1]/AU:.4f}, {mars_pos[2]/AU:.4f}]")
    
    # Step 3: Design Earth-Moon transfer
    print("\nStep 3: Designing Earth-Moon transfer trajectory")
    
    # Parking orbit around Earth (low Earth orbit)
    leo_alt = 300e3  # 300 km altitude
    leo_radius = EARTH_RADIUS + leo_alt
    leo_speed = np.sqrt(EARTH_MU / leo_radius)
    
    # Initial position (in Earth's frame)
    # Position that optimizes for the Moon transfer
    # For simplicity, start in the direction of the Moon
    init_dir = moon_pos_rel / np.linalg.norm(moon_pos_rel)
    init_pos_rel = leo_radius * init_dir
    
    # Velocity perpendicular to position vector for circular orbit
    init_vel_mag = np.sqrt(EARTH_MU / leo_radius)
    init_vel_dir = np.array([-init_dir[1], init_dir[0], 0])  # 90 degrees from position
    init_vel_rel = init_vel_mag * init_vel_dir
    
    # Convert to solar system frame
    init_pos = earth_pos + init_pos_rel
    init_vel = earth_vel + init_vel_rel
    
    # Moon encounter after ~3 days (typical transit time)
    moon_encounter_date = mission_start + timedelta(days=3)
    tof_earth_moon = (moon_encounter_date - mission_start).total_seconds()
    
    # For demonstration, using realistic delta-V values based on typical mission parameters
    # Earth departure delta-V (typical LEO to trans-lunar injection)
    dv_earth_departure = 3100  # m/s, realistic value
    
    # Calculate spacecraft velocity after Earth departure burn
    departure_vel_dir = (moon_pos - init_pos) / np.linalg.norm(moon_pos - init_pos)
    v1_trans_moon = init_vel + departure_vel_dir * dv_earth_departure
    
    print(f"Earth departure delta-V: {dv_earth_departure/1000:.2f} km/s (typical LEO to TLI)")
    print(f"Moon encounter date: {moon_encounter_date.strftime('%Y-%m-%d')}")
    
    # Propagate Earth-Moon trajectory using numerical integration
    earth_moon_result = propagate_orbit(init_pos, v1_trans_moon, tof_earth_moon, SUN_MU)
    if earth_moon_result:
        earth_moon_traj = earth_moon_result['trajectory']
        
        # Get arrival position and velocity at the Moon
        sc_pos_at_moon = earth_moon_traj[-1, 0:3]
        v2_arr_moon = earth_moon_traj[-1, 3:6]
        
        # Distance to Moon at "arrival"
        arrival_error = np.linalg.norm(sc_pos_at_moon - moon_pos)
        print(f"Distance to Moon at arrival: {arrival_error/1000:.2f} km")
        
        # Adjust to Moon's position for the next calculations
        sc_pos_at_moon = moon_pos
    else:
        # Fallback to simple approximation
        print("Using approximation for Earth-Moon transfer...")
        tof_fraction = 0.9  # 90% of the transfer time
        sc_pos_at_moon = moon_pos
        # Estimate arrival velocity using vis-viva equation
        r1 = np.linalg.norm(init_pos - earth_pos)
        r2 = np.linalg.norm(moon_pos - earth_pos)
        a = (r1 + r2) / 2  # Semi-major axis of transfer orbit
        v2_arr_moon = np.sqrt(SUN_MU * (2/np.linalg.norm(moon_pos) - 1/a)) * (moon_pos / np.linalg.norm(moon_pos))
    
    # Step 4: Design Lunar Gravity Assist
    print("\nStep 4: Calculating lunar gravity assist")
    
    # Spacecraft velocity at Moon encounter
    sc_vel_at_moon = v2_arr_moon
    
    # Spacecraft velocity relative to the Moon
    v_rel_moon = sc_vel_at_moon - moon_vel
    v_rel_moon_mag = np.linalg.norm(v_rel_moon)
    
    # Parameters for gravity assist
    # Periapsis distance of the hyperbolic trajectory around the Moon
    periapsis_alt = 200e3  # 200 km altitude flyby (realistic)
    rp = MOON_RADIUS + periapsis_alt
    
    # Realistic lunar flyby parameters
    v_inf = v_rel_moon_mag  # Velocity at infinity (approach velocity)
    print(f"Approach velocity (v_inf): {v_inf/1000:.2f} km/s")
    
    # Semi-major axis of the hyperbola
    a_hyp = -MOON_MU / (v_inf**2) 
    
    # Eccentricity of the hyperbola
    e_hyp = 1 - rp / abs(a_hyp)
    
    # Realistic deflection angle for lunar gravity assist
    # For lunar flybys, typical bend angles are 30-60 degrees
    delta = np.radians(45)  # 45 degrees is a realistic bend angle
    
    # Calculate unit vector perpendicular to the orbital plane
    h_vec = np.cross(moon_pos, v_rel_moon)
    h_unit = h_vec / np.linalg.norm(h_vec)
    
    # Create a rotation matrix around this perpendicular vector
    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)
    
    # Rodrigues rotation formula
    K = np.array([
        [0, -h_unit[2], h_unit[1]],
        [h_unit[2], 0, -h_unit[0]],
        [-h_unit[1], h_unit[0], 0]
    ])
    
    R = np.eye(3) + sin_delta * K + (1 - cos_delta) * (K @ K)
    
    # Apply rotation to get outgoing velocity relative to Moon
    v_out_rel_moon = R @ v_rel_moon
    
    # Final velocity after gravity assist (in solar system frame)
    v_out_sun = v_out_rel_moon + moon_vel
    
    # Velocity change due to gravity assist
    dv_gravity_assist = np.linalg.norm(v_out_sun - sc_vel_at_moon)
    
    print(f"Spacecraft velocity before assist: {np.linalg.norm(sc_vel_at_moon)/1000:.2f} km/s")
    print(f"Spacecraft velocity after assist: {np.linalg.norm(v_out_sun)/1000:.2f} km/s")
    print(f"Velocity change from gravity assist: {dv_gravity_assist/1000:.2f} km/s")
    print(f"Bend angle: {np.degrees(delta):.1f} degrees")
    
    # Step 5: Calculate Moon-Mars transfer
    print("\nStep 5: Calculating Moon-Mars transfer trajectory")
    
    tof_moon_mars = (mars_arrival_date - moon_encounter_date).total_seconds()
    
    # For a realistic mission, we'll use typical delta-V values
    # Midcourse correction after lunar gravity assist
    dv_after_assist = 500  # m/s, realistic for trajectory correction
    
    # Calculate transfer velocity to Mars
    mars_transfer_dir = (mars_pos - moon_pos) / np.linalg.norm(mars_pos - moon_pos)
    v1_trans_mars = v_out_sun + mars_transfer_dir * dv_after_assist
    
    # Mars orbit insertion delta-V (typical value)
    dv_mars_insertion = 1000  # m/s, realistic for Mars orbit insertion
    
    print(f"Moon-Mars transfer delta-V: {dv_after_assist/1000:.2f} km/s")
    print(f"Mars orbit insertion delta-V: {dv_mars_insertion/1000:.2f} km/s")
    
    # Total mission delta-V
    total_dv = dv_earth_departure + dv_after_assist + dv_mars_insertion
    print(f"\nTotal mission delta-V: {total_dv/1000:.2f} km/s")
    
    # Propagate Moon-Mars trajectory using numerical integration
    moon_mars_result = propagate_orbit(moon_pos, v1_trans_mars, tof_moon_mars, SUN_MU)
    if moon_mars_result:
        moon_mars_traj = moon_mars_result['trajectory']
        # Final position at Mars
        final_pos = moon_mars_traj[-1, 0:3]
        mars_arrival_error = np.linalg.norm(final_pos - mars_pos)
        print(f"Distance to Mars at arrival: {mars_arrival_error/1000:.2f} km")
    
    # Step 6: Compare with direct Earth-Mars transfer
    print("\nStep 6: Comparing with direct Earth-Mars transfer")
    
    # Direct Earth-Mars transfer using typical values
    tof_direct = (mars_arrival_date - mission_start).total_seconds()
    
    # Typical delta-V values for direct Earth-Mars transfer
    dv_direct_departure = 3800  # m/s, typical for direct trans-Mars injection
    dv_direct_arrival = 1500    # m/s, typical for Mars orbit insertion
    
    # Total delta-V for direct transfer
    total_direct_dv = dv_direct_departure + dv_direct_arrival
    
    print(f"Direct transfer departure delta-V: {dv_direct_departure/1000:.2f} km/s")
    print(f"Direct transfer arrival delta-V: {dv_direct_arrival/1000:.2f} km/s")
    print(f"Total direct transfer delta-V: {total_direct_dv/1000:.2f} km/s")
    
    # Calculate direct transfer trajectory
    direct_transfer_dir = (mars_pos - earth_pos) / np.linalg.norm(mars_pos - earth_pos)
    v1_direct = earth_vel + direct_transfer_dir * dv_direct_departure
    
    # Propagate direct Earth-Mars trajectory
    direct_result = propagate_orbit(earth_pos, v1_direct, tof_direct, SUN_MU)
    if direct_result:
        direct_traj = direct_result['trajectory']
    
    # Calculate delta-V savings
    dv_savings = total_direct_dv - total_dv
    if dv_savings > 0:
        print(f"\nDelta-V savings with lunar gravity assist: {dv_savings/1000:.2f} km/s")
        print(f"Percentage savings: {(dv_savings/total_direct_dv)*100:.1f}%")
    else:
        print(f"\nAdditional delta-V cost with lunar gravity assist: {-dv_savings/1000:.2f} km/s")
        print(f"Percentage penalty: {(-dv_savings/total_direct_dv)*100:.1f}%")
        print("Note: This penalty is expected, as gravity assists typically extend mission duration")
        print("but can provide benefits like reduced propellant mass or increased payload capacity.")
    
    # Step 7: Visualize trajectories
    print("\nStep 7: Visualizing trajectories")
    
    # Create a figure for the trajectory
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    setup_solar_system_plot(ax, max_radius=1.8, title="Earth-Moon-Mars Mission with Lunar Gravity Assist")
    
    # Plot Earth and Mars orbits
    plot_planet_orbit(ax, "Earth", color=PLANETS['Earth']['color'])
    plot_planet_orbit(ax, "Mars", color=PLANETS['Mars']['color'])
    
    # Plot Earth, Moon, and Mars positions
    ax.scatter(earth_pos[0]/AU, earth_pos[1]/AU, earth_pos[2]/AU, 
              color=PLANETS['Earth']['color'], s=100, label="Earth at departure")
    ax.scatter(moon_pos[0]/AU, moon_pos[1]/AU, moon_pos[2]/AU, 
              color='gray', s=50, label="Moon at encounter")
    ax.scatter(mars_pos[0]/AU, mars_pos[1]/AU, mars_pos[2]/AU, 
              color=PLANETS['Mars']['color'], s=100, label="Mars at arrival")
    
    # Plot trajectories using numerically integrated paths
    if 'earth_moon_traj' in locals():
        plot_trajectory_3d(ax, earth_moon_traj[:, 0:3]/AU, color='blue', linewidth=2, 
                         label="Earth to Moon")
    
    if 'moon_mars_traj' in locals():
        plot_trajectory_3d(ax, moon_mars_traj[:, 0:3]/AU, color='red', linewidth=2, 
                         label="Moon to Mars (with Gravity Assist)")
    
    if 'direct_traj' in locals():
        plot_trajectory_3d(ax, direct_traj[:, 0:3]/AU, color='green', linestyle='--', 
                         linewidth=1.5, label="Direct Earth-Mars (for comparison)")
    
    # Add legend and save
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "lunar_gravity_assist_mission.png"))
    
    # Create comparison chart
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = ['Direct Transfer', 'Lunar Gravity Assist']
    dvs = [total_direct_dv/1000, total_dv/1000]  # Convert to km/s
    
    ax.bar(methods, dvs, color=['green', 'blue'])
    ax.set_ylabel('Total Delta-V (km/s)')
    ax.set_title('Comparison of Delta-V Requirements')
    
    # Add value labels on top of bars
    for i, v in enumerate(dvs):
        ax.text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    # Add savings annotation
    if dv_savings > 0:
        savings_text = f"Savings: {dv_savings/1000:.2f} km/s ({(dv_savings/total_direct_dv)*100:.1f}%)"
        box_color = "lightgreen"
    else:
        savings_text = f"Additional cost: {-dv_savings/1000:.2f} km/s ({(-dv_savings/total_direct_dv)*100:.1f}%)"
        box_color = "lightsalmon"
        
    plt.annotate(savings_text, xy=(0.5, 0.9), xycoords='axes fraction', 
                ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc=box_color, alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "gravity_assist_comparison.png"))
    
    print(f"\nVisualizations saved to the '{OUTPUT_DIR}' directory.")
    print("\nNote: The results shown are based on simplified calculations and are for educational purposes only.")
    print("Real mission design involves much more detailed analysis and optimization.")

if __name__ == "__main__":
    main() 