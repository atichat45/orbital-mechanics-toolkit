#!/usr/bin/env python3
"""
Download SPICE kernels for orbital mechanics calculations.
"""

import os
import requests
from tqdm import tqdm

def download_kernel(url, output_path):
    """Download a SPICE kernel from a URL."""
    # Create directories if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
                desc=os.path.basename(url),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
        
        print(f"Downloaded to {output_path}")
        return True
    else:
        print(f"Failed to download {url}")
        return False

def main():
    """Download all required kernels."""
    kernel_dir = "data/kernels"
    os.makedirs(kernel_dir, exist_ok=True)
    
    # List of kernels to download
    kernels = [
        ("https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls", f"{kernel_dir}/naif0012.tls"),
        ("https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc", f"{kernel_dir}/pck00010.tpc"),
        ("https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp", f"{kernel_dir}/de440.bsp"),
    ]
    
    for url, path in kernels:
        if not os.path.exists(path):
            download_kernel(url, path)
        else:
            print(f"Skipping {path} (already exists)")

if __name__ == "__main__":
    main() 