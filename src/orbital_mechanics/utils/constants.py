"""
Physical and Astronomical Constants

This module provides constants used throughout the orbital mechanics package.
"""

import numpy as np

# Universal Constants
G = 6.67430e-11  # Gravitational constant, m^3/kg/s^2

# Time Constants
SECOND = 1.0
MINUTE = 60.0 * SECOND
HOUR = 60.0 * MINUTE
DAY = 24.0 * HOUR
YEAR = 365.25 * DAY

# Distance Constants
METER = 1.0
KILOMETER = 1000.0 * METER
AU = 149.597870700e9 * METER  # Astronomical Unit, m

# Mass Constants
KILOGRAM = 1.0

# Celestial Body Properties
# Sun
SUN_MASS = 1.989e30  # kg
SUN_RADIUS = 695700e3  # m
SUN_MU = G * SUN_MASS  # m^3/s^2

# Earth
EARTH_MASS = 5.972e24  # kg
EARTH_RADIUS = 6371e3  # m
EARTH_MU = G * EARTH_MASS  # m^3/s^2
EARTH_ORBITAL_PERIOD = 365.256 * DAY  # s
EARTH_SEMI_MAJOR_AXIS = AU  # m
EARTH_ECCENTRICITY = 0.0167
EARTH_J2 = 1.08263e-3

# Moon
MOON_MASS = 7.342e22  # kg
MOON_RADIUS = 1737.4e3  # m
MOON_MU = G * MOON_MASS  # m^3/s^2
MOON_ORBITAL_PERIOD = 27.321582 * DAY  # s
MOON_SEMI_MAJOR_AXIS = 384400e3  # m
MOON_ECCENTRICITY = 0.0549

# Mars
MARS_MASS = 6.417e23  # kg
MARS_RADIUS = 3389.5e3  # m
MARS_MU = G * MARS_MASS  # m^3/s^2
MARS_SEMI_MAJOR_AXIS = 1.524 * AU  # m
MARS_ECCENTRICITY = 0.0934
MARS_ORBITAL_PERIOD = 686.980 * DAY  # s

# Venus
VENUS_MASS = 4.867e24  # kg
VENUS_RADIUS = 6051.8e3  # m
VENUS_MU = G * VENUS_MASS  # m^3/s^2
VENUS_SEMI_MAJOR_AXIS = 0.723332 * AU  # m
VENUS_ECCENTRICITY = 0.006772
VENUS_ORBITAL_PERIOD = 224.701 * DAY  # s

# Mercury
MERCURY_MASS = 3.301e23  # kg
MERCURY_RADIUS = 2439.7e3  # m
MERCURY_MU = G * MERCURY_MASS  # m^3/s^2
MERCURY_SEMI_MAJOR_AXIS = 0.387098 * AU  # m
MERCURY_ECCENTRICITY = 0.205630
MERCURY_ORBITAL_PERIOD = 87.9691 * DAY  # s

# Jupiter
JUPITER_MASS = 1.898e27  # kg
JUPITER_RADIUS = 69911e3  # m
JUPITER_MU = G * JUPITER_MASS  # m^3/s^2
JUPITER_SEMI_MAJOR_AXIS = 5.2038 * AU  # m
JUPITER_ECCENTRICITY = 0.0489
JUPITER_ORBITAL_PERIOD = 11.8618 * YEAR  # s

# Saturn
SATURN_MASS = 5.683e26  # kg
SATURN_RADIUS = 58232e3  # m
SATURN_MU = G * SATURN_MASS  # m^3/s^2
SATURN_SEMI_MAJOR_AXIS = 9.5370 * AU  # m
SATURN_ECCENTRICITY = 0.0565
SATURN_ORBITAL_PERIOD = 29.4571 * YEAR  # s

# Uranus
URANUS_MASS = 8.681e25  # kg
URANUS_RADIUS = 25362e3  # m
URANUS_MU = G * URANUS_MASS  # m^3/s^2
URANUS_SEMI_MAJOR_AXIS = 19.189 * AU  # m
URANUS_ECCENTRICITY = 0.0457
URANUS_ORBITAL_PERIOD = 84.0205 * YEAR  # s

# Neptune
NEPTUNE_MASS = 1.024e26  # kg
NEPTUNE_RADIUS = 24622e3  # m
NEPTUNE_MU = G * NEPTUNE_MASS  # m^3/s^2
NEPTUNE_SEMI_MAJOR_AXIS = 30.070 * AU  # m
NEPTUNE_ECCENTRICITY = 0.0113
NEPTUNE_ORBITAL_PERIOD = 164.8 * YEAR  # s

# Dictionary of planet properties for easier access
PLANETS = {
    'Mercury': {
        'mass': MERCURY_MASS,
        'radius': MERCURY_RADIUS,
        'mu': MERCURY_MU,
        'semi_major_axis': MERCURY_SEMI_MAJOR_AXIS,
        'eccentricity': MERCURY_ECCENTRICITY,
        'orbital_period': MERCURY_ORBITAL_PERIOD,
        'color': 'gray'
    },
    'Venus': {
        'mass': VENUS_MASS,
        'radius': VENUS_RADIUS,
        'mu': VENUS_MU,
        'semi_major_axis': VENUS_SEMI_MAJOR_AXIS,
        'eccentricity': VENUS_ECCENTRICITY,
        'orbital_period': VENUS_ORBITAL_PERIOD,
        'color': 'gold'
    },
    'Earth': {
        'mass': EARTH_MASS,
        'radius': EARTH_RADIUS,
        'mu': EARTH_MU,
        'semi_major_axis': EARTH_SEMI_MAJOR_AXIS,
        'eccentricity': EARTH_ECCENTRICITY,
        'orbital_period': EARTH_ORBITAL_PERIOD,
        'color': 'blue'
    },
    'Mars': {
        'mass': MARS_MASS,
        'radius': MARS_RADIUS,
        'mu': MARS_MU,
        'semi_major_axis': MARS_SEMI_MAJOR_AXIS,
        'eccentricity': MARS_ECCENTRICITY,
        'orbital_period': MARS_ORBITAL_PERIOD,
        'color': 'red'
    },
    'Jupiter': {
        'mass': JUPITER_MASS,
        'radius': JUPITER_RADIUS,
        'mu': JUPITER_MU,
        'semi_major_axis': JUPITER_SEMI_MAJOR_AXIS,
        'eccentricity': JUPITER_ECCENTRICITY,
        'orbital_period': JUPITER_ORBITAL_PERIOD,
        'color': 'orange'
    },
    'Saturn': {
        'mass': SATURN_MASS,
        'radius': SATURN_RADIUS,
        'mu': SATURN_MU,
        'semi_major_axis': SATURN_SEMI_MAJOR_AXIS,
        'eccentricity': SATURN_ECCENTRICITY,
        'orbital_period': SATURN_ORBITAL_PERIOD,
        'color': 'khaki'
    },
    'Uranus': {
        'mass': URANUS_MASS,
        'radius': URANUS_RADIUS,
        'mu': URANUS_MU,
        'semi_major_axis': URANUS_SEMI_MAJOR_AXIS,
        'eccentricity': URANUS_ECCENTRICITY,
        'orbital_period': URANUS_ORBITAL_PERIOD,
        'color': 'skyblue'
    },
    'Neptune': {
        'mass': NEPTUNE_MASS,
        'radius': NEPTUNE_RADIUS,
        'mu': NEPTUNE_MU,
        'semi_major_axis': NEPTUNE_SEMI_MAJOR_AXIS,
        'eccentricity': NEPTUNE_ECCENTRICITY,
        'orbital_period': NEPTUNE_ORBITAL_PERIOD,
        'color': 'blue'
    }
} 