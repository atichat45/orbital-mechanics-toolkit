#!/usr/bin/env python3
"""
Lambert Transfer Analysis Script

This script performs an analysis of Lambert's problem solutions for
interplanetary transfers, including porkchop plot generation for
launch window analysis.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.orbital_mechanics.core.lambert import solve_lambert
from src.orbital_mechanics.data.spice_interface import SpiceInterface
from src.orbital_mechanics.utils.constants import PLANETS, AU, DAY
from src.orbital_mechanics.visualization.orbit_plotting import plot_porkchop, plot_trajectory_3d
from src.orbital_mechanics.analysis.interplanetary import porkchop_analysis, find_optimal_launch_window

def run_lambert_analysis():
    """
    Run a comprehensive analysis of Lambert transfers for interplanetary missions.
    """
    print("Lambert Transfer Analysis")
    print("========================")
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize SPICE interface and load kernels
    print("Initializing SPICE interface...")
    spice = SpiceInterface()
    spice.load_kernels()
    
    try:
        # Define analysis parameters
        origin = 'Earth'
        destination = 'Mars'
        
        # Time window for analysis
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        # Time of flight range (days)
        min_tof = 100
        max_tof = 500
        
        print(f"\nAnalyzing {origin}-{destination} transfers from {start_date.strftime('%Y-%m-%d')} "
              f"to {end_date.strftime('%Y-%m-%d')}")
        print(f"Time of flight range: {min_tof}-{max_tof} days")
        
        # Generate porkchop plot data
        print("\nGenerating porkchop plot data (this may take a few minutes)...")
        porkchop_data = porkchop_analysis(
            spice=spice,
            origin=origin,
            destination=destination,
            start_date=start_date,
            end_date=end_date,
            min_tof=min_tof,
            max_tof=max_tof,
            departure_steps=20,
            tof_steps=20
        )
        
        # Create porkchop plot
        print("Creating porkchop plot...")
        fig, ax = plt.subplots(figsize=(14, 10))
        plot_porkchop(
            ax=ax,
            data=porkchop_data,
            title=f"{origin}-{destination} Transfer Opportunities ({start_date.year}-{end_date.year})",
            c3_levels=np.arange(5, 40, 2.5),
            dv_arrival_levels=np.arange(1, 10, 0.5)
        )
        plt.savefig(os.path.join(output_dir, f"{origin.lower()}_{destination.lower()}_porkchop.png"), 
                   dpi=300, bbox_inches='tight')
        print(f"Porkchop plot saved as '{origin.lower()}_{destination.lower()}_porkchop.png'")
        
        # Find optimal launch windows
        print("\nFinding optimal launch windows...")
        windows = find_optimal_launch_window(
            porkchop_data=porkchop_data,
            num_windows=3,
            min_separation_days=60
        )
        
        # Print and visualize optimal launch windows
        print("\nOptimal Launch Windows:")
        print("----------------------")
        
        for i, window in enumerate(windows):
            print(f"\nWindow {i+1}:")
            print(f"  Departure Date: {window['departure_date'].strftime('%Y-%m-%d')}")
            print(f"  Arrival Date: {window['arrival_date'].strftime('%Y-%m-%d')}")
            print(f"  Time of Flight: {window['tof_days']:.1f} days")
            print(f"  Departure C3: {window['c3']:.2f} km²/s²")
            print(f"  Arrival ΔV: {window['arrival_dv']:.2f} km/s")
            print(f"  Total ΔV: {window['total_dv']:.2f} km/s")
            
            # Plot the optimal trajectory
            print(f"  Generating trajectory visualization...")
            
            # Get positions at departure and arrival
            r1 = spice.get_position(
                target=origin, 
                observer='SUN', 
                datetime_obj=window['departure_date']
            )
            
            r2 = spice.get_position(
                target=destination, 
                observer='SUN', 
                datetime_obj=window['arrival_date']
            )
            
            # Solve Lambert's problem for this specific transfer
            tof_seconds = window['tof_days'] * DAY
            v1, v2 = solve_lambert(
                r1=r1, 
                r2=r2, 
                tof=tof_seconds, 
                mu=PLANETS['Sun']['mu'],
                clockwise=False
            )
            
            # Generate trajectory points
            num_points = 100
            trajectory = np.zeros((num_points, 3))
            
            # Create trajectory points by propagating from departure
            for j in range(num_points):
                t = j * tof_seconds / (num_points - 1)
                pos = spice.propagate_kepler(r1, v1, t, PLANETS['Sun']['mu'])
                trajectory[j] = pos
            
            # Plot trajectory
            fig = plt.figure(figsize=(14, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot origin and destination orbits
            departure_orbit = []
            arrival_orbit = []
            
            # Generate points for origin orbit
            for angle in np.linspace(0, 2*np.pi, 100):
                pos = spice.get_position(
                    target=origin, 
                    observer='SUN', 
                    datetime_obj=window['departure_date'] + timedelta(days=angle/(2*np.pi)*365.25)
                )
                departure_orbit.append(pos)
            
            # Generate points for destination orbit
            for angle in np.linspace(0, 2*np.pi, 100):
                pos = spice.get_position(
                    target=destination, 
                    observer='SUN', 
                    datetime_obj=window['arrival_date'] + timedelta(days=angle/(2*np.pi)*365.25)
                )
                arrival_orbit.append(pos)
            
            departure_orbit = np.array(departure_orbit)
            arrival_orbit = np.array(arrival_orbit)
            
            # Plot orbits
            ax.plot(
                departure_orbit[:, 0]/AU, 
                departure_orbit[:, 1]/AU, 
                departure_orbit[:, 2]/AU, 
                'b-', 
                alpha=0.5, 
                label=f"{origin} Orbit"
            )
            
            ax.plot(
                arrival_orbit[:, 0]/AU, 
                arrival_orbit[:, 1]/AU, 
                arrival_orbit[:, 2]/AU, 
                'r-', 
                alpha=0.5, 
                label=f"{destination} Orbit"
            )
            
            # Plot Sun
            ax.scatter([0], [0], [0], color='yellow', s=200, label='Sun')
            
            # Plot departure and arrival positions
            ax.scatter(
                [r1[0]/AU], [r1[1]/AU], [r1[2]/AU], 
                color='blue', s=100, 
                label=f"{origin} at Departure"
            )
            
            ax.scatter(
                [r2[0]/AU], [r2[1]/AU], [r2[2]/AU], 
                color='red', s=100, 
                label=f"{destination} at Arrival"
            )
            
            # Plot transfer trajectory
            ax.plot(
                trajectory[:, 0]/AU, 
                trajectory[:, 1]/AU, 
                trajectory[:, 2]/AU, 
                'g-', 
                linewidth=2, 
                label='Transfer Trajectory'
            )
            
            # Configure plot
            max_limit = max(
                np.max(np.abs(departure_orbit/AU)), 
                np.max(np.abs(arrival_orbit/AU))
            ) * 1.1
            
            ax.set_xlim(-max_limit, max_limit)
            ax.set_ylim(-max_limit, max_limit)
            ax.set_zlim(-max_limit, max_limit)
            
            ax.set_xlabel('X (AU)')
            ax.set_ylabel('Y (AU)')
            ax.set_zlabel('Z (AU)')
            
            ax.set_title(f"Optimal {origin}-{destination} Transfer\n"
                        f"Departure: {window['departure_date'].strftime('%Y-%m-%d')}, "
                        f"Arrival: {window['arrival_date'].strftime('%Y-%m-%d')}, "
                        f"ToF: {window['tof_days']:.1f} days")
            
            # Add details as text
            details = (f"Transfer Details:\n"
                     f"ΔV Departure: {window['departure_dv']:.2f} km/s\n"
                     f"ΔV Arrival: {window['arrival_dv']:.2f} km/s\n"
                     f"Total ΔV: {window['total_dv']:.2f} km/s\n"
                     f"C3: {window['c3']:.2f} km²/s²")
            
            ax.text2D(0.05, 0.05, details, transform=ax.transAxes,
                    fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
            
            # Improve view angle
            ax.view_init(elev=30, azim=45)
            
            # Add legend
            ax.legend(loc='upper right', fontsize=10)
            
            # Save figure
            plt.savefig(
                os.path.join(
                    output_dir, 
                    f"{origin.lower()}_{destination.lower()}_optimal_transfer_{i+1}.png"
                ), 
                dpi=300, 
                bbox_inches='tight'
            )
            print(f"  Trajectory plot saved as '{origin.lower()}_{destination.lower()}_optimal_transfer_{i+1}.png'")
            
            plt.close()
        
        # Summarize results and recommendations
        print("\nMission Planning Recommendations:")
        print("---------------------------------")
        
        # Sort windows by total delta-V
        windows_by_dv = sorted(windows, key=lambda w: w['total_dv'])
        best_window = windows_by_dv[0]
        
        print(f"Most efficient launch window: {best_window['departure_date'].strftime('%Y-%m-%d')}")
        print(f"  Arrival: {best_window['arrival_date'].strftime('%Y-%m-%d')}")
        print(f"  Total ΔV: {best_window['total_dv']:.2f} km/s")
        print(f"  Time of Flight: {best_window['tof_days']:.1f} days")
        
        # Find the quickest transfer that's within 10% of optimal delta-V
        viable_windows = [w for w in windows if w['total_dv'] <= best_window['total_dv'] * 1.1]
        if viable_windows:
            quickest = min(viable_windows, key=lambda w: w['tof_days'])
            if quickest['departure_date'] != best_window['departure_date']:
                print(f"\nQuickest viable transfer (within 10% of optimal ΔV):")
                print(f"  Departure: {quickest['departure_date'].strftime('%Y-%m-%d')}")
                print(f"  Arrival: {quickest['arrival_date'].strftime('%Y-%m-%d')}")
                print(f"  Total ΔV: {quickest['total_dv']:.2f} km/s")
                print(f"  Time of Flight: {quickest['tof_days']:.1f} days")
                print(f"  Time saved: {best_window['tof_days'] - quickest['tof_days']:.1f} days")
                print(f"  Extra ΔV required: {quickest['total_dv'] - best_window['total_dv']:.2f} km/s")
        
        # Save results to CSV
        results_df = pd.DataFrame(windows)
        csv_path = os.path.join(output_dir, f"{origin.lower()}_{destination.lower()}_launch_windows.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\nDetailed results saved to '{csv_path}'")
        
    finally:
        # Unload SPICE kernels
        spice.unload_kernels()
        print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    run_lambert_analysis() 