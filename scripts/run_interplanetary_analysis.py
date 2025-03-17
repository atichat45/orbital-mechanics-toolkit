#!/usr/bin/env python3
"""
Interplanetary Transfer Analysis Script

This script performs a comprehensive analysis of Hohmann transfers
from Earth to all other planets in the Solar System.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.orbital_mechanics.analysis.interplanetary import (
    compute_all_planet_transfers,
    analyze_interplanetary_transfers_summary
)
from src.orbital_mechanics.visualization.orbit_plotting import (
    setup_solar_system_plot,
    plot_planet_orbit,
    plot_trajectory_3d,
    plot_hohmann_transfer
)
from src.orbital_mechanics.utils.constants import PLANETS, AU

def run_interplanetary_analysis():
    """
    Run a comprehensive analysis of interplanetary transfers.
    """
    print("Interplanetary Hohmann Transfer Analysis")
    print("======================================")
    
    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Compute transfers from Earth to all planets
    print("Calculating transfer parameters for all planets...")
    results_df = compute_all_planet_transfers(origin='Earth')
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    create_visualizations(results_df, output_dir)
    
    # Analyze results
    print("\nAnalyzing results...")
    analysis = analyze_interplanetary_transfers_summary(results_df)
    
    # Generate report
    generate_report(analysis, output_dir)
    
    print("\nAnalysis completed successfully!")

def create_visualizations(results_df, output_dir):
    """
    Create visualizations of the interplanetary transfers.
    
    Args:
        results_df: DataFrame with transfer parameters
        output_dir: Directory to save visualizations
    """
    # 1. Create 3D plot of all transfers
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up the solar system plot
    max_radius = max(planet['semi_major_axis'] for planet in PLANETS.values()) * 1.1
    setup_solar_system_plot(ax, max_radius=max_radius, 
                          title='Hohmann Transfer Trajectories from Earth to Other Planets')
    
    # Plot Earth's orbit
    plot_planet_orbit(ax, 'Earth', position=0, z_offset=0)
    
    # Plot each planet's orbit and transfer trajectory
    for i, row in results_df.iterrows():
        planet_name = row['target']
        
        # Plot the planet's orbit
        z_offset = PLANETS[planet_name]['semi_major_axis'] * 0.02  # Small offset for visibility
        plot_planet_orbit(ax, planet_name, position=np.pi, z_offset=z_offset)
        
        # Plot the transfer trajectory
        trajectory = row['numerical_trajectory']
        
        # Add slight vertical offset for better visibility
        trajectory_with_offset = trajectory.copy()
        trajectory_with_offset[:, 2] += z_offset
        
        plot_trajectory_3d(ax, trajectory_with_offset, 
                           color=PLANETS[planet_name]['color'],
                           label=f"Transfer to {planet_name}")
    
    # Improve legend - separate into two legends for better clarity
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
    
    # Place the legends in better positions
    ax.legend(planet_handles, planet_labels, loc='upper left', 
             title="Celestial Bodies", frameon=True, framealpha=0.9)
    
    # Add second legend for transfers
    second_legend = ax.figure.legend(transfer_handles, transfer_labels, 
                                    loc='upper right', title="Transfer Trajectories", 
                                    frameon=True, framealpha=0.9)
    ax.figure.add_artist(second_legend)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'hohmann_transfers_3d.png'), dpi=300, bbox_inches='tight')
    print("3D plot saved as 'hohmann_transfers_3d.png'")
    
    # 2. Create a zoomed-in plot of inner planet transfers
    create_inner_planets_visualization(results_df, output_dir)
    
    # 3. Create comparative visualizations
    create_comparative_visualization(results_df, output_dir)

def create_inner_planets_visualization(results_df, output_dir):
    """
    Create a visualization focusing on transfers to inner planets.
    
    Args:
        results_df: DataFrame with transfer parameters
        output_dir: Directory to save visualizations
    """
    # Filter for inner planets
    inner_planets = ['Mercury', 'Venus', 'Mars']
    inner_df = results_df[results_df['target'].isin(inner_planets)]
    
    # Create figure
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up the solar system plot
    max_radius = PLANETS['Mars']['semi_major_axis'] * 1.2
    setup_solar_system_plot(ax, max_radius=max_radius, 
                          title='Hohmann Transfers from Earth to Inner Planets')
    
    # Plot Earth's orbit
    plot_planet_orbit(ax, 'Earth', position=0, z_offset=0)
    
    # Plot each inner planet's orbit and transfer trajectory
    for i, row in inner_df.iterrows():
        planet_name = row['target']
        
        # Plot the planet's orbit
        z_offset = PLANETS[planet_name]['semi_major_axis'] * 0.02  # Small offset for visibility
        plot_planet_orbit(ax, planet_name, position=np.pi, z_offset=z_offset)
        
        # Plot the transfer trajectory
        trajectory = row['numerical_trajectory']
        
        # Add slight vertical offset for better visibility
        trajectory_with_offset = trajectory.copy()
        trajectory_with_offset[:, 2] += z_offset
        
        plot_trajectory_3d(ax, trajectory_with_offset, 
                         color=PLANETS[planet_name]['color'],
                         label=f"Transfer to {planet_name}")
        
        # Add text with key metrics
        info_text = (f"{planet_name}:\n"
                    f"ΔV: {row['delta_v']/1000:.1f} km/s\n"
                    f"Time: {row['tof_days']:.1f} days\n"
                    f"Path: {row['analytical_path_length']/AU:.2f} AU")
        
        # Position the text in a suitable location
        text_x = PLANETS[planet_name]['semi_major_axis'] * np.cos(np.pi/4) / AU
        text_y = PLANETS[planet_name]['semi_major_axis'] * np.sin(np.pi/4) / AU
        text_z = z_offset * 3 / AU
        
        # Add textbox with information
        ax.text(text_x, text_y, text_z, info_text, color=PLANETS[planet_name]['color'],
               fontweight='bold', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Improve legend - separate into two legends for better clarity
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
    
    # Place the legends in better positions
    ax.legend(planet_handles, planet_labels, loc='upper left', 
             title="Celestial Bodies", frameon=True, framealpha=0.9)
    
    # Add second legend for transfers
    second_legend = ax.figure.legend(transfer_handles, transfer_labels, 
                                   loc='upper right', title="Transfer Trajectories", 
                                   frameon=True, framealpha=0.9)
    ax.figure.add_artist(second_legend)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'inner_planets_transfers.png'), dpi=300, bbox_inches='tight')
    print("Inner planets plot saved as 'inner_planets_transfers.png'")

def create_comparative_visualization(results_df, output_dir):
    """
    Create comparative visualizations of transfer parameters.
    
    Args:
        results_df: DataFrame with transfer parameters
        output_dir: Directory to save visualizations
    """
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
    })
    
    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    
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
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'hohmann_transfers_comparison.png'), dpi=300, bbox_inches='tight')
    print("Comparison plots saved as 'hohmann_transfers_comparison.png'")

def generate_report(analysis, output_dir):
    """
    Generate a report of the interplanetary transfer analysis.
    
    Args:
        analysis: Dictionary with analysis results
        output_dir: Directory to save the report
    """
    # Print summary to console
    print("\n===== INTERPLANETARY HOHMANN TRANSFERS SUMMARY =====\n")
    print(f"Total planets analyzed: {analysis['total_planets']}\n")
    
    print("Path Length Analysis:")
    print(f"  Shortest path: {analysis['path_length']['shortest']['planet']} "
          f"({analysis['path_length']['shortest']['length_au']:.2f} AU)")
    print(f"  Longest path: {analysis['path_length']['longest']['planet']} "
          f"({analysis['path_length']['longest']['length_au']:.2f} AU)")
    print(f"  Ratio of longest to shortest: {analysis['path_length']['ratio']:.2f}\n")
    
    print("Time of Flight Analysis:")
    print(f"  Shortest time: {analysis['time_of_flight']['shortest']['planet']} "
          f"({analysis['time_of_flight']['shortest']['days']:.2f} days, "
          f"{analysis['time_of_flight']['shortest']['years']:.2f} years)")
    print(f"  Longest time: {analysis['time_of_flight']['longest']['planet']} "
          f"({analysis['time_of_flight']['longest']['days']:.2f} days, "
          f"{analysis['time_of_flight']['longest']['years']:.2f} years)")
    print(f"  Ratio of longest to shortest: {analysis['time_of_flight']['ratio']:.2f}\n")
    
    print("Delta-V Analysis:")
    print(f"  Lowest delta-V: {analysis['delta_v']['lowest']['planet']} "
          f"({analysis['delta_v']['lowest']['value']:.2f} km/s)")
    print(f"  Highest delta-V: {analysis['delta_v']['highest']['planet']} "
          f"({analysis['delta_v']['highest']['value']:.2f} km/s)")
    print(f"  Ratio of highest to lowest: {analysis['delta_v']['ratio']:.2f}\n")
    
    # Print detailed planetary data
    df = analysis['detailed_results']
    print("Detailed Planetary Data:")
    print("Planet     Path (AU)    Time (days)     Time (years)    ΔV (km/s)   ")
    print("----------------------------------------------------------------------")
    for i, row in df.iterrows():
        print(f"{row['target']:<10} {row['analytical_path_length_au']:<12.2f} "
              f"{row['tof_days']:<15.2f} {row['tof_years']:<15.2f} "
              f"{row['delta_v']/1000:<12.2f}")
    
    print("\nCorrelation Analysis:\n")
    print("Correlation Matrix:")
    print(analysis['correlation_matrix'])
    
    # Save detailed results to CSV
    csv_path = os.path.join(output_dir, 'hohmann_transfers_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to '{csv_path}'")

if __name__ == "__main__":
    run_interplanetary_analysis() 