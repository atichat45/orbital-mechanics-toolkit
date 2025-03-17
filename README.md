# Orbital Mechanics Toolkit

A comprehensive Python library for orbital mechanics calculations, trajectory design, and mission analysis. This toolkit provides accurate and efficient implementations of fundamental orbital mechanics algorithms and visualization tools for space mission planning and analysis.

## 🚀 Features

- **Core Orbital Mechanics**: Keplerian elements, coordinate transformations, and orbital maneuvers
- **Lambert's Problem Solver**: For trajectory design and interplanetary transfers
- **Hohmann Transfer Calculator**: Optimized algorithms for transfer orbits
- **Gravity Assist Planning**: Tools for designing lunar and planetary flyby maneuvers
- **Porkchop Plot Generation**: Launch window analysis for interplanetary missions
- **SPICE Integration**: Interface with NASA's SPICE toolkit for high-precision ephemeris
- **3D Visualization**: Interactive plotting of orbits, trajectories, and celestial bodies
- **Numerical Propagation**: Accurate orbit propagation with various integration methods

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/atichat45/orbital-mechanics-toolkit.git
cd orbital-mechanics-toolkit

# Install the package
pip install -e .
```

## 🔧 Usage Example

```python
from orbital_mechanics.core.orbital_elements import cartesian_to_keplerian
from orbital_mechanics.core.lambert import solve_lambert
from orbital_mechanics.utils.constants import SUN_MU
from orbital_mechanics.visualization.orbit_plotting import plot_orbit_3d

# Calculate a Mars transfer trajectory
earth_pos = [...]  # Earth position vector
mars_pos = [...]   # Mars position vector
tof = 210 * 86400  # Transfer time in seconds (210 days)

v1, v2 = solve_lambert(earth_pos, mars_pos, tof, SUN_MU)

# Convert to orbital elements
elements = cartesian_to_keplerian(earth_pos, v1, SUN_MU)
```

## 📚 Documentation

Comprehensive documentation is available in the `/docs` directory, including:
- API reference
- Tutorial notebooks
- Example mission scenarios
- Theoretical background

## 🛠️ Project Structure

- `src/orbital_mechanics/core/`: Core orbital mechanics algorithms
- `src/orbital_mechanics/analysis/`: Mission analysis tools
- `src/orbital_mechanics/visualization/`: 2D and 3D plotting utilities
- `src/orbital_mechanics/utils/`: Constants and utility functions
- `src/orbital_mechanics/data/`: SPICE interface and data handling
- `examples/`: Example scripts and use cases
- `tests/`: Unit and integration tests

## 🔬 Scientific Accuracy

This toolkit implements algorithms based on established astrodynamics textbooks and papers, with validation against industry-standard tools where possible.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
# orbital-mechanics-toolkit
# orbital-mechanics-toolkit
# orbital-mechanics-toolkit
# orbital-mechanics-toolkit
