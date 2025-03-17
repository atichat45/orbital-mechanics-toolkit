"""
Mars Mission Analysis Example

This script demonstrates how to use the orbital mechanics toolkit to analyze 
a Mars mission, including:
1. Creating a porkchop plot to find optimal launch windows
2. Visualizing the transfer trajectory
3. Calculating delta-V requirements
4. Comparing different transfer options
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from mpl_toolkits.mplot3d import Axes3D

from orbital_mechanics.analysis.interplanetary import porkchop_analysis, find_optimal_launch_window
from orbital_mechanics.core.lambert import solve_lambert
from orbital_mechanics.visualization.orbit_plotting import (
    setup_solar_system_plot, 
    plot_planet_orbit, 
    plot_trajectory_3d,
    plot_porkchop
)
from orbital_mechanics.utils.constants import AU, DAY, YEAR, SUN_MU, PLANETS

# Output directory for saving results
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # 1. Define mission parameters
    origin = "Earth"
    destination = "Mars"
    
    # Launch window to consider (2022-2024)
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 1, 1)
    
    # Time of flight range to consider (days)
    min_tof = 120
    max_tof = 300
    
    print(f"Analyzing {origin}-{destination} transfers from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Time of flight range: {min_tof}-{max_tof} days")
    
    # 2. Generate porkchop plot data
    print("Generating porkchop plot data (this may take a moment)...")
    porkchop_data = porkchop_analysis(
        origin, 
        destination, 
        start_date,
        end_date,
        min_tof_days=min_tof,
        max_tof_days=max_tof,
        departure_steps=40, 
        tof_steps=40
    )
    
    # 3. Find the optimal launch window
    print("Finding optimal launch window...")
    optimal = find_optimal_launch_window(
        origin, 
        destination, 
        start_date,
        end_date,
        min_tof_days=min_tof,
        max_tof_days=max_tof,
        departure_steps=60, 
        tof_steps=60
    )
    
    # Print optimal transfer details
    print("\nOptimal Transfer Parameters:")
    print(f"Departure Date: {optimal['departure_date'].strftime('%Y-%m-%d')}")
    print(f"Arrival Date: {optimal['arrival_date'].strftime('%Y-%m-%d')}")
    print(f"Time of Flight: {optimal['tof_days']:.1f} days")
    print(f"Delta-V: {optimal['delta_v']/1000:.2f} km/s")
    print(f"C3: {optimal['c3']/1e6:.2f} km²/s²")
    
    # 4. Create porkchop plot
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    plot_porkchop(
        ax, 
        porkchop_data, 
        f"{origin}-{destination} Transfer Opportunities (2022-2024)"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{origin.lower()}_to_{destination.lower()}_porkchop.png"))
    
    # 5. Visualize the optimal transfer trajectory
    visualize_transfer(origin, destination, optimal)
    
    # 6. Compare different transfer types
    compare_transfers(origin, destination, optimal['departure_date'])
    
    print("\nAnalysis complete. Results saved to the 'output' directory.")

def visualize_transfer(origin, destination, optimal):
    """Visualize the transfer trajectory for the optimal launch window."""
    # Setup the figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    setup_solar_system_plot(ax, max_radius=1.8, title=f"Optimal {origin}-{destination} Transfer Trajectory")
    
    # Plot planet orbits
    plot_planet_orbit(ax, origin, color=PLANETS[origin]['color'])
    plot_planet_orbit(ax, destination, color=PLANETS[destination]['color'])
    
    # Get state vectors for planets at departure and arrival
    # Origin planet at departure
    r1 = PLANETS[origin]['semi_major_axis']
    v1_circular = np.sqrt(SUN_MU / r1)
    
    # Calculate departure and arrival angles
    dep_date = optimal['departure_date']
    arr_date = optimal['arrival_date']
    
    angle1 = (dep_date - datetime(2000, 1, 1)).total_seconds() / (PLANETS[origin]['orbital_period'] + 1e-10) * 2 * np.pi
    r1_vec = np.array([r1 * np.cos(angle1), r1 * np.sin(angle1), 0]) / AU
    v1_vec = np.array([-v1_circular * np.sin(angle1), v1_circular * np.cos(angle1), 0])
    
    # Position of destination at arrival
    r2 = PLANETS[destination]['semi_major_axis']
    v2_circular = np.sqrt(SUN_MU / r2)
    
    angle2 = (arr_date - datetime(2000, 1, 1)).total_seconds() / (PLANETS[destination]['orbital_period'] + 1e-10) * 2 * np.pi
    r2_vec = np.array([r2 * np.cos(angle2), r2 * np.sin(angle2), 0]) / AU
    v2_vec = np.array([-v2_circular * np.sin(angle2), v2_circular * np.cos(angle2), 0])
    
    # Plot planet positions at departure and arrival
    ax.scatter(r1_vec[0], r1_vec[1], r1_vec[2], color=PLANETS[origin]['color'], s=100, label=f"{origin} at departure")
    ax.scatter(r2_vec[0], r2_vec[1], r2_vec[2], color=PLANETS[destination]['color'], s=100, label=f"{destination} at arrival")
    
    # Calculate transfer trajectory using Lambert's solution
    r1_vec_m = r1_vec * AU  # Convert back to meters for calculation
    r2_vec_m = r2_vec * AU
    tof = optimal['tof_days'] * DAY  # Time of flight in seconds
    
    try:
        v1_transfer, v2_transfer = solve_lambert(r1_vec_m, r2_vec_m, tof, SUN_MU)
        
        # Generate trajectory points for visualization
        points = 100
        positions = np.zeros((points, 3))
        
        # Simple numerical propagation for visualization
        dt = tof / (points - 1)
        pos = r1_vec_m
        vel = v1_transfer
        
        for i in range(points):
            positions[i] = pos / AU  # Store position in AU
            # Simple Euler integration (for visualization only)
            r_mag = np.linalg.norm(pos)
            acc = -SUN_MU * pos / (r_mag**3)
            vel = vel + acc * dt
            pos = pos + vel * dt
        
        # Plot the transfer trajectory
        plot_trajectory_3d(ax, positions, color='red', linewidth=2, 
                         label=f"Transfer Trajectory\n({optimal['tof_days']:.1f} days)")
    
    except Exception as e:
        print(f"Error calculating transfer trajectory: {e}")
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{origin.lower()}_to_{destination.lower()}_trajectory.png"))

def compare_transfers(origin, destination, departure_date, tof_range=[120, 180, 240, 300]):
    """Compare different transfer times and their delta-V requirements."""
    
    results = []
    
    for tof_days in tof_range:
        arrival_date = departure_date + timedelta(days=tof_days)
        
        # Origin planet at departure
        r1 = PLANETS[origin]['semi_major_axis']
        v1_circular = np.sqrt(SUN_MU / r1)
        
        angle1 = (departure_date - datetime(2000, 1, 1)).total_seconds() / (PLANETS[origin]['orbital_period'] + 1e-10) * 2 * np.pi
        r1_vec = np.array([r1 * np.cos(angle1), r1 * np.sin(angle1), 0])
        v1_vec = np.array([-v1_circular * np.sin(angle1), v1_circular * np.cos(angle1), 0])
        
        # Position of destination at arrival
        r2 = PLANETS[destination]['semi_major_axis']
        v2_circular = np.sqrt(SUN_MU / r2)
        
        angle2 = (arrival_date - datetime(2000, 1, 1)).total_seconds() / (PLANETS[destination]['orbital_period'] + 1e-10) * 2 * np.pi
        r2_vec = np.array([r2 * np.cos(angle2), r2 * np.sin(angle2), 0])
        v2_vec = np.array([-v2_circular * np.sin(angle2), v2_circular * np.cos(angle2), 0])
        
        # Calculate transfer using Lambert's solution
        tof = tof_days * DAY  # Time of flight in seconds
        
        try:
            v1_transfer, v2_transfer = solve_lambert(r1_vec, r2_vec, tof, SUN_MU)
            
            # Calculate delta-Vs
            dv_departure = np.linalg.norm(v1_transfer - v1_vec) / 1000  # km/s
            dv_arrival = np.linalg.norm(v2_vec - v2_transfer) / 1000  # km/s
            dv_total = dv_departure + dv_arrival
            
            # C3 energy
            c3 = np.linalg.norm(v1_transfer)**2 - 2*SUN_MU/np.linalg.norm(r1_vec)
            
            results.append({
                'TOF (days)': tof_days,
                'Departure ΔV (km/s)': dv_departure,
                'Arrival ΔV (km/s)': dv_arrival,
                'Total ΔV (km/s)': dv_total,
                'C3 (km²/s²)': c3 / 1e6
            })
            
        except Exception as e:
            print(f"Error calculating transfer for TOF={tof_days} days: {e}")
    
    # Create comparison table and plot
    if results:
        # Create table
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = [[r['TOF (days)'], 
                      f"{r['Departure ΔV (km/s)']:.2f}", 
                      f"{r['Arrival ΔV (km/s)']:.2f}", 
                      f"{r['Total ΔV (km/s)']:.2f}",
                      f"{r['C3 (km²/s²)']:.2f}"] for r in results]
        
        table = ax.table(cellText=table_data,
                      colLabels=['TOF (days)', 'Departure ΔV (km/s)', 'Arrival ΔV (km/s)', 
                                'Total ΔV (km/s)', 'C3 (km²/s²)'],
                      loc='center', cellLoc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.title(f"Comparison of {origin}-{destination} Transfers\nDeparture: {departure_date.strftime('%Y-%m-%d')}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{origin.lower()}_to_{destination.lower()}_comparison.png"))
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        tofs = [r['TOF (days)'] for r in results]
        dep_dvs = [r['Departure ΔV (km/s)'] for r in results]
        arr_dvs = [r['Arrival ΔV (km/s)'] for r in results]
        
        x = np.arange(len(tofs))
        width = 0.35
        
        ax.bar(x - width/2, dep_dvs, width, label='Departure ΔV')
        ax.bar(x + width/2, arr_dvs, width, label='Arrival ΔV')
        
        ax.set_xlabel('Time of Flight (days)')
        ax.set_ylabel('Delta-V (km/s)')
        ax.set_title(f'Delta-V Requirements for Different {origin}-{destination} Transfer Times')
        ax.set_xticks(x)
        ax.set_xticklabels(tofs)
        ax.legend()
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{origin.lower()}_to_{destination.lower()}_dv_comparison.png"))

if __name__ == "__main__":
    main() 