#!/usr/bin/env python3
"""
Script to specifically check Mars data in SPICE.
"""

import spiceypy as spice
import numpy as np
from datetime import datetime

def main():
    # Load the kernels
    print("Loading SPICE kernels...")
    spice.furnsh("data/kernels/naif0012.tls")
    spice.furnsh("data/kernels/de440.bsp")
    spice.furnsh("data/kernels/pck00010.tpc")
    
    # Try a simple query on Mars
    try:
        # Try for year 2000
        et_2000 = spice.str2et("2000-01-01T00:00:00")
        mars_state = spice.spkezr("499", et_2000, "J2000", "NONE", "10")
        print(f"Successfully retrieved Mars state for 2000-01-01:")
        print(f"Position: {mars_state[0][:3]}")
        print(f"Velocity: {mars_state[0][3:]}")
    except Exception as e:
        print(f"Error getting Mars state for 2000: {e}")
    
    # Try different dates to find working range
    test_years = [1950, 1970, 1990, 2000, 2010, 2020, 2030, 2050]
    
    print("\nTesting Mars data availability for different years:")
    print("--------------------------------------------------")
    
    for year in test_years:
        date_str = f"{year}-01-01T00:00:00"
        try:
            et = spice.str2et(date_str)
            state = spice.spkezr("MARS", et, "J2000", "NONE", "SUN")
            print(f"{date_str}: SUCCESS - Mars data available")
        except Exception as e:
            print(f"{date_str}: FAILED - {str(e)}")
    
    # Unload kernels
    spice.unload("data/kernels/naif0012.tls")
    spice.unload("data/kernels/de440.bsp")
    spice.unload("data/kernels/pck00010.tpc")
    
if __name__ == "__main__":
    main() 