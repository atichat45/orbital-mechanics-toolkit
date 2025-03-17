"""
Interplanetary Transfer Analysis

This module provides functions for analyzing interplanetary transfers,
including Hohmann transfers, low-thrust trajectories, and launch windows.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ..utils.constants import AU, DAY, YEAR, SUN_MU, PLANETS
from ..core.hohmann import hohmann_transfer_analytical, propagate_orbit_numerical
from ..core.lambert import solve_lambert, multi_rev_lambert
from ..core.orbital_elements import cartesian_to_keplerian, keplerian_to_cartesian

def compute_interplanetary_hohmann(origin, destination):
    """
    Compute parameters for a Hohmann transfer between two planets.
    
    Args:
        origin: Name of the origin planet
        destination: Name of the destination planet
        
    Returns:
        Dictionary with transfer parameters
    """
    # Get the orbital parameters of origin and destination
    r1 = PLANETS[origin]['semi_major_axis']
    r2 = PLANETS[destination]['semi_major_axis']
    
    # Compute the Hohmann transfer parameters
    hohmann_params = hohmann_transfer_analytical(r1, r2, SUN_MU)
    
    # Calculate the time of flight in days
    tof_days = hohmann_params['tof'] / DAY
    
    # Calculate the path length in astronomical units
    path_length_au = hohmann_params['path_length'] / AU
    
    return {
        'origin': origin,
        'destination': destination,
        'r1': r1,
        'r2': r2,
        'a_transfer': hohmann_params['a_transfer'],
        'e_transfer': hohmann_params['e_transfer'],
        'delta_v_departure': hohmann_params['delta_v_departure'],
        'delta_v_arrival': hohmann_params['delta_v_arrival'],
        'delta_v_total': hohmann_params['delta_v_total'],
        'tof': hohmann_params['tof'],
        'tof_days': tof_days,
        'path_length': hohmann_params['path_length'],
        'path_length_au': path_length_au,
        'c3': hohmann_params['c3']
    }

def compute_all_planet_transfers(origin='Earth'):
    """
    Compute Hohmann transfers from a specified origin to all other planets.
    
    Args:
        origin: Name of the origin planet (default: 'Earth')
        
    Returns:
        DataFrame with transfer parameters for all planets
    """
    results = []
    
    for planet in PLANETS.keys():
        # Skip the origin planet
        if planet == origin:
            continue
        
        # Compute the transfer parameters
        params = compute_interplanetary_hohmann(origin, planet)
        
        # Store the results
        results.append({
            'target': planet,
            'a_target': PLANETS[planet]['semi_major_axis'],
            'e_target': PLANETS[planet]['eccentricity'],
            'a_transfer': params['a_transfer'],
            'e_transfer': params['e_transfer'],
            'delta_v': params['delta_v_total'],
            'departure_dv': params['delta_v_departure'],
            'arrival_dv': params['delta_v_arrival'],
            'tof_days': params['tof_days'],
            'tof_years': params['tof_days'] / 365.25,
            'analytical_path_length': params['path_length'],
            'analytical_path_length_au': params['path_length_au'],
            'c3': params['c3']
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Add numerical trajectories and path lengths
    for i, row in df.iterrows():
        # Get planetary positions for a Hohmann transfer
        # (Simplified - assuming circular coplanar orbits)
        r1 = PLANETS[origin]['semi_major_axis']
        r2 = row['a_target']
        
        # Departure position (assuming planets at 0 deg)
        departure_pos = np.array([r1, 0, 0])
        departure_vel = np.array([0, np.sqrt(SUN_MU / r1), 0])
        
        # Velocity needed for the Hohmann transfer
        v1_transfer = departure_vel[1] + row['departure_dv']
        transfer_vel = np.array([0, v1_transfer, 0])
        
        # Propagate the transfer orbit
        transfer_trajectory = propagate_orbit_numerical(
            departure_pos, transfer_vel, row['tof_days'] * DAY, SUN_MU
        )
        
        # Store the trajectory and numerical path length
        df.at[i, 'numerical_trajectory'] = transfer_trajectory['trajectory']
        df.at[i, 'numerical_path_length'] = transfer_trajectory['path_length']
        df.at[i, 'numerical_path_length_au'] = transfer_trajectory['path_length'] / AU
    
    return df

def porkchop_analysis(origin, destination, start_date, end_date, 
                     min_tof_days=10, max_tof_days=500, 
                     departure_steps=30, tof_steps=30):
    """
    Generate data for a porkchop plot for transfers between two planets.
    
    Args:
        origin: Name of the origin planet
        destination: Name of the destination planet
        start_date: Start date for departure window (datetime)
        end_date: End date for departure window (datetime)
        min_tof_days: Minimum time of flight in days
        max_tof_days: Maximum time of flight in days
        departure_steps: Number of departure dates to evaluate
        tof_steps: Number of time-of-flight values to evaluate
        
    Returns:
        Dictionary with porkchop data
    """
    # Generate departure dates and times of flight
    departure_dates = [start_date + i * (end_date - start_date) / (departure_steps - 1) 
                      for i in range(departure_steps)]
    
    tof_values = np.linspace(min_tof_days, max_tof_days, tof_steps) * DAY
    
    # Initialize result arrays
    departure_date_arr = np.array(departure_dates)
    tof_arr = np.array(tof_values) / DAY
    delta_v_grid = np.zeros((len(departure_dates), len(tof_values)))
    c3_grid = np.zeros((len(departure_dates), len(tof_values)))
    arrival_date_grid = np.zeros((len(departure_dates), len(tof_values)), dtype=object)
    
    # Compute transfer for each combination
    for i, dep_date in enumerate(departure_dates):
        for j, tof in enumerate(tof_values):
            # Calculate arrival date
            arr_date = dep_date + timedelta(seconds=float(tof))
            arrival_date_grid[i, j] = arr_date
            
            # Get state vectors for planets at departure and arrival
            # In a real implementation, these would come from ephemeris data
            # Here we're using a simplified model with circular coplanar orbits
            
            # Origin planet at departure
            r1 = PLANETS[origin]['semi_major_axis']
            v1_circular = np.sqrt(SUN_MU / r1)
            
            # Angular position based on date (simplified)
            # Add a small value to orbital period to prevent division by zero
            angle1 = (dep_date - datetime(2000, 1, 1)).total_seconds() / (PLANETS[origin]['orbital_period'] + 1e-10) * 2 * np.pi
            r1_vec = np.array([r1 * np.cos(angle1), r1 * np.sin(angle1), 0])
            v1_vec = np.array([-v1_circular * np.sin(angle1), v1_circular * np.cos(angle1), 0])
            
            # Destination planet at arrival
            r2 = PLANETS[destination]['semi_major_axis']
            v2_circular = np.sqrt(SUN_MU / r2)
            
            # Angular position based on date (simplified)
            # Add a small value to orbital period to prevent division by zero
            angle2 = (arr_date - datetime(2000, 1, 1)).total_seconds() / (PLANETS[destination]['orbital_period'] + 1e-10) * 2 * np.pi
            r2_vec = np.array([r2 * np.cos(angle2), r2 * np.sin(angle2), 0])
            v2_vec = np.array([-v2_circular * np.sin(angle2), v2_circular * np.cos(angle2), 0])
            
            try:
                # Solve Lambert's problem
                v1_transfer, v2_transfer = solve_lambert(r1_vec, r2_vec, tof, SUN_MU)
                
                # Calculate departure delta-v
                delta_v_dep = np.linalg.norm(v1_transfer - v1_vec)
                
                # Calculate arrival delta-v
                delta_v_arr = np.linalg.norm(v2_vec - v2_transfer)
                
                # Total delta-v
                delta_v_total = delta_v_dep + delta_v_arr
                
                # C3 (characteristic energy)
                c3 = np.linalg.norm(v1_transfer)**2 - 2*SUN_MU/np.linalg.norm(r1_vec)
                
                # Store values
                delta_v_grid[i, j] = delta_v_total / 1000  # km/s
                c3_grid[i, j] = c3 / 1e6  # km²/s²
                
            except np.linalg.LinAlgError as e:
                # Handle linear algebra errors (e.g., singular matrix)
                print(f"Linear algebra error at dep={dep_date}, tof={tof/DAY:.1f} days: {e}")
                delta_v_grid[i, j] = np.nan
                c3_grid[i, j] = np.nan
            except Exception as e:
                # If Lambert solver fails, set values to NaN
                print(f"Lambert solver error at dep={dep_date}, tof={tof/DAY:.1f} days: {e}")
                delta_v_grid[i, j] = np.nan
                c3_grid[i, j] = np.nan
    
    return {
        'departure_dates': departure_date_arr,
        'tof_days': tof_arr,
        'arrival_dates': arrival_date_grid,
        'delta_v': delta_v_grid,
        'c3': c3_grid
    }

def find_optimal_launch_window(origin, destination, start_date, end_date, 
                              min_tof_days=10, max_tof_days=500, 
                              departure_steps=50, tof_steps=50):
    """
    Find the optimal launch window for an interplanetary transfer.
    
    Args:
        origin: Name of the origin planet
        destination: Name of the destination planet
        start_date: Start date for departure window (datetime)
        end_date: End date for departure window (datetime)
        min_tof_days: Minimum time of flight in days
        max_tof_days: Maximum time of flight in days
        departure_steps: Number of departure dates to evaluate
        tof_steps: Number of time-of-flight values to evaluate
        
    Returns:
        Dictionary with optimal transfer parameters
    """
    # Generate porkchop data
    porkchop_data = porkchop_analysis(
        origin, destination, start_date, end_date,
        min_tof_days, max_tof_days, departure_steps, tof_steps
    )
    
    # Find minimum delta-v transfer
    min_delta_v = np.nanmin(porkchop_data['delta_v'])
    min_idx = np.unravel_index(np.nanargmin(porkchop_data['delta_v']), porkchop_data['delta_v'].shape)
    
    # Extract optimal parameters
    optimal_departure_date = porkchop_data['departure_dates'][min_idx[0]]
    optimal_tof_days = porkchop_data['tof_days'][min_idx[1]]
    optimal_arrival_date = porkchop_data['arrival_dates'][min_idx]
    optimal_c3 = porkchop_data['c3'][min_idx]
    
    return {
        'origin': origin,
        'destination': destination,
        'departure_date': optimal_departure_date,
        'arrival_date': optimal_arrival_date,
        'tof_days': optimal_tof_days,
        'delta_v': min_delta_v * 1000,  # m/s
        'c3': optimal_c3 * 1e6,  # m²/s²
        'porkchop_data': porkchop_data
    }

def analyze_interplanetary_transfers_summary(results_df):
    """
    Generate a comprehensive summary of interplanetary transfer analysis.
    
    Args:
        results_df: DataFrame with transfer parameters for all planets
        
    Returns:
        Dictionary with summary statistics and analysis
    """
    # Find minimum and maximum values for key metrics
    min_path = results_df.loc[results_df['analytical_path_length_au'].idxmin()]
    max_path = results_df.loc[results_df['analytical_path_length_au'].idxmax()]
    
    min_time = results_df.loc[results_df['tof_days'].idxmin()]
    max_time = results_df.loc[results_df['tof_days'].idxmax()]
    
    min_dv = results_df.loc[results_df['delta_v'].idxmin()]
    max_dv = results_df.loc[results_df['delta_v'].idxmax()]
    
    # Calculate ratios
    path_ratio = max_path['analytical_path_length_au'] / min_path['analytical_path_length_au']
    time_ratio = max_time['tof_days'] / min_time['tof_days']
    dv_ratio = max_dv['delta_v'] / min_dv['delta_v']
    
    # Calculate correlations
    correlation_matrix = results_df[[
        'a_target', 'analytical_path_length', 'numerical_path_length', 
        'tof_days', 'delta_v'
    ]].corr()
    
    return {
        'total_planets': len(results_df),
        'path_length': {
            'shortest': {
                'planet': min_path['target'],
                'length_au': min_path['analytical_path_length_au']
            },
            'longest': {
                'planet': max_path['target'],
                'length_au': max_path['analytical_path_length_au']
            },
            'ratio': path_ratio
        },
        'time_of_flight': {
            'shortest': {
                'planet': min_time['target'],
                'days': min_time['tof_days'],
                'years': min_time['tof_days'] / 365.25
            },
            'longest': {
                'planet': max_time['target'],
                'days': max_time['tof_days'],
                'years': max_time['tof_days'] / 365.25
            },
            'ratio': time_ratio
        },
        'delta_v': {
            'lowest': {
                'planet': min_dv['target'],
                'value': min_dv['delta_v'] / 1000  # km/s
            },
            'highest': {
                'planet': max_dv['target'],
                'value': max_dv['delta_v'] / 1000  # km/s
            },
            'ratio': dv_ratio
        },
        'correlation_matrix': correlation_matrix,
        'detailed_results': results_df
    } 