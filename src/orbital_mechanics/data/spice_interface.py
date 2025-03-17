"""
SPICE Interface

This module provides functions for working with SPICE ephemeris data.
"""

import os
import numpy as np
import spiceypy as spice
from datetime import datetime, timedelta

class SpiceInterface:
    """
    Interface for SPICE ephemeris system.
    
    This class handles loading SPICE kernels, retrieving ephemeris data,
    and converting between time systems.
    """
    
    def __init__(self, kernel_directory="data/kernels"):
        """
        Initialize the SPICE interface.
        
        Args:
            kernel_directory: Directory containing SPICE kernels
        """
        self.kernel_directory = kernel_directory
        self.loaded_kernels = []
        
    def load_kernels(self, kernel_files=None):
        """
        Load SPICE kernels from the specified files.
        
        Args:
            kernel_files: List of kernel filenames to load (relative to kernel_directory)
                          If None, load standard set of kernels
        """
        if kernel_files is None:
            # Default set of kernels
            kernel_files = [
                "naif0012.tls",  # Leap seconds kernel
                "de440.bsp",     # Planetary ephemeris kernel
                "pck00010.tpc"   # Planetary constants kernel
            ]
            
        # Load each kernel
        for kernel_file in kernel_files:
            kernel_path = os.path.join(self.kernel_directory, kernel_file)
            if os.path.exists(kernel_path):
                spice.furnsh(kernel_path)
                self.loaded_kernels.append(kernel_path)
                print(f"Loaded SPICE kernel: {kernel_path}")
            else:
                print(f"Warning: SPICE kernel not found: {kernel_path}")
    
    def unload_kernels(self):
        """
        Unload all previously loaded SPICE kernels.
        """
        for kernel_path in self.loaded_kernels:
            spice.unload(kernel_path)
        self.loaded_kernels = []
        print("Unloaded all SPICE kernels")
    
    def datetime_to_et(self, dt):
        """
        Convert a Python datetime to SPICE ephemeris time (ET).
        
        Args:
            dt: Python datetime object
            
        Returns:
            Ephemeris time (seconds past J2000)
        """
        time_string = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        return spice.str2et(time_string)
    
    def et_to_datetime(self, et):
        """
        Convert SPICE ephemeris time (ET) to Python datetime.
        
        Args:
            et: Ephemeris time (seconds past J2000)
            
        Returns:
            Python datetime object
        """
        time_string = spice.et2utc(et, "ISOC", 6)
        return datetime.strptime(time_string, "%Y-%m-%dT%H:%M:%S.%f")
    
    def get_state(self, target_body, observer_body, dt=None, et=None, frame="ECLIPJ2000"):
        """
        Get the state (position and velocity) of a body relative to an observer.
        
        Args:
            target_body: SPICE ID or name of target body
            observer_body: SPICE ID or name of observer body
            dt: Python datetime for the query (if provided, et is ignored)
            et: Ephemeris time for the query (seconds past J2000)
            frame: Reference frame
            
        Returns:
            Dictionary containing state information
        """
        if dt is not None:
            et = self.datetime_to_et(dt)
        
        if et is None:
            raise ValueError("Either dt or et must be provided")
        
        try:
            # Get state vector
            state, lt = spice.spkezr(target_body, et, frame, "NONE", observer_body)
            
            # Extract position and velocity
            position = np.array(state[0:3])
            velocity = np.array(state[3:6])
            
            return {
                'position': position,
                'velocity': velocity,
                'et': et,
                'lt': lt,
                'datetime': self.et_to_datetime(et)
            }
        except Exception as e:
            print(f"Error getting state for {target_body} relative to {observer_body}: {str(e)}")
            return None
    
    def get_position(self, target_body, observer_body, dt=None, et=None, frame="ECLIPJ2000"):
        """
        Get the position of a body relative to an observer.
        
        Args:
            target_body: SPICE ID or name of target body
            observer_body: SPICE ID or name of observer body
            dt: Python datetime for the query (if provided, et is ignored)
            et: Ephemeris time for the query (seconds past J2000)
            frame: Reference frame
            
        Returns:
            Position vector (3D numpy array)
        """
        state = self.get_state(target_body, observer_body, dt, et, frame)
        if state:
            return state['position']
        return None
    
    def check_coverage(self, target_body, observer_body, start_dt, end_dt, step=timedelta(days=30)):
        """
        Check if SPICE data is available for a given body over a time range.
        
        Args:
            target_body: SPICE ID or name of target body
            observer_body: SPICE ID or name of observer body
            start_dt: Start datetime
            end_dt: End datetime
            step: Time step for checking
            
        Returns:
            Dictionary with coverage information
        """
        current_dt = start_dt
        coverage_data = {
            'start_dt': start_dt,
            'end_dt': end_dt,
            'covered_dates': [],
            'missing_dates': [],
            'error_messages': []
        }
        
        while current_dt <= end_dt:
            try:
                position = self.get_position(target_body, observer_body, current_dt)
                if position is not None:
                    coverage_data['covered_dates'].append(current_dt)
                else:
                    coverage_data['missing_dates'].append(current_dt)
                    coverage_data['error_messages'].append(f"No position data at {current_dt}")
            except spice.stypes.SpiceyError as e:
                coverage_data['missing_dates'].append(current_dt)
                coverage_data['error_messages'].append(f"SPICE error at {current_dt}: {str(e)}")
            except Exception as e:
                coverage_data['missing_dates'].append(current_dt)
                coverage_data['error_messages'].append(f"Error at {current_dt}: {str(e)}")
            
            current_dt += step
        
        coverage_data['has_full_coverage'] = len(coverage_data['missing_dates']) == 0
        coverage_data['coverage_percentage'] = (len(coverage_data['covered_dates']) / 
                                              (len(coverage_data['covered_dates']) + len(coverage_data['missing_dates'])) * 100)
        
        return coverage_data

    def download_kernel(self, kernel_url, output_path=None):
        """
        Download a SPICE kernel from a URL.
        
        Args:
            kernel_url: URL of the kernel to download
            output_path: Path to save the kernel (if None, use kernel_directory)
            
        Returns:
            Path to the downloaded kernel
        """
        import requests
        from tqdm import tqdm
        
        # Set output path
        if output_path is None:
            if not os.path.exists(self.kernel_directory):
                os.makedirs(self.kernel_directory)
            output_path = os.path.join(self.kernel_directory, os.path.basename(kernel_url))
        
        # Download the kernel
        response = requests.get(kernel_url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                    desc=os.path.basename(kernel_url),
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
            
            print(f"Downloaded kernel to {output_path}")
            return output_path
        else:
            print(f"Failed to download kernel from {kernel_url}")
            return None 