#!/usr/bin/env python3
"""
Check SPICE kernel coverage for different bodies.
"""

import spiceypy as spice
import os
from datetime import datetime, timedelta

def load_spice_kernels():
    """Load necessary SPICE kernels."""
    print("Loading SPICE kernels...")
    
    kernels_dir = 'data/kernels'
    lsk_path = os.path.join(kernels_dir, 'naif0012.tls')
    spk_path = os.path.join(kernels_dir, 'de440.bsp')
    pck_path = os.path.join(kernels_dir, 'pck00010.tpc')
    
    # Load kernels
    spice.furnsh(lsk_path)
    spice.furnsh(spk_path)
    spice.furnsh(pck_path)
    
    return [lsk_path, spk_path, pck_path]

def unload_spice_kernels(kernel_paths):
    """Unload SPICE kernels."""
    print("Unloading SPICE kernels...")
    for kernel in kernel_paths:
        spice.unload(kernel)

def check_coverage(body_id, observer_id=10, start_year=2000, end_year=2025):
    """
    Check SPICE data coverage for a specific body.
    
    Args:
        body_id: SPICE ID of the target body
        observer_id: SPICE ID of the observer body (default: Sun)
        start_year: Beginning year to check
        end_year: Ending year to check
    """
    body_name = get_body_name(body_id)
    observer_name = get_body_name(observer_id)
    
    print(f"\nChecking coverage for {body_name} observed from {observer_name}:")
    
    # Check by month for each year
    for year in range(start_year, end_year + 1):
        print(f"  {year}: ", end="")
        has_data = []
        
        for month in range(1, 13):
            # Create a date string for the middle of the month
            date_str = f"{year}-{month:02d}-15T12:00:00"
            
            try:
                # Convert to ephemeris time
                et = spice.str2et(date_str)
                
                # Try to get the state vector
                state, lt = spice.spkezr(
                    str(body_id), 
                    et, 
                    "ECLIPJ2000", 
                    "NONE", 
                    str(observer_id)
                )
                
                # If we get here, data exists
                has_data.append(month)
                print(f"{month:02d} ", end="")
                
            except Exception as e:
                print("-- ", end="")
        
        if has_data:
            print(f"({len(has_data)} months covered)")
        else:
            print("(No coverage)")

def get_body_name(body_id):
    """Get the name of a body from its SPICE ID."""
    try:
        return spice.bodc2n(body_id)
    except:
        return f"Body-{body_id}"

def main():
    try:
        # Load SPICE kernels
        kernel_paths = load_spice_kernels()
        
        # List of bodies to check
        bodies = [
            {"id": 10, "name": "Sun"},
            {"id": 399, "name": "Earth"},
            {"id": 499, "name": "Mars"},
            {"id": 599, "name": "Jupiter"},
            {"id": 699, "name": "Saturn"}
        ]
        
        # Print kernel info
        print("\nLoaded kernels:")
        for kernel in kernel_paths:
            print(f"  {kernel}")
        
        # Check coverage for each body as observed from Sun
        for body in bodies:
            if body["id"] != 10:  # Skip Sun observing itself
                check_coverage(body["id"])
        
        # Also check Earth-Mars coverage specifically
        print("\nChecking coverage for Mars observed from Earth:")
        check_coverage(499, 399)  # Mars as seen from Earth
        
    except Exception as e:
        print(f"Error checking coverage: {e}")
    finally:
        # Unload SPICE kernels
        unload_spice_kernels(kernel_paths)

if __name__ == "__main__":
    main() 