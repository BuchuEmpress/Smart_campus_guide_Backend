"""
Location Utilities Module

This module provides pure Python utility functions for geographic calculations
and coordinate manipulation. No external API calls are made here.

Functions:
    - calculate_distance: Calculate distance between two points using Haversine formula
    - get_cardinal_direction: Get compass direction (N, NE, E, etc.)
    - calculate_bearing: Calculate bearing in degrees (0-360)
    - format_distance: Format distance for human-readable display
    - format_time: Format time duration for human-readable display
    - validate_coordinates: Validate if coordinates are within valid ranges
    - is_on_campus: Check if coordinates are within campus boundaries


"""

import math
from typing import Tuple, Dict, Optional


# Earth's radius in meters (used for Haversine formula)
EARTH_RADIUS_METERS = 6371000


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two geographic points using Haversine formula.
    
    The Haversine formula determines the great-circle distance between two points
    on a sphere given their longitudes and latitudes. This is accurate for most
    terrestrial distances.
    
    Args:
        lat1: Latitude of first point in decimal degrees
        lon1: Longitude of first point in decimal degrees
        lat2: Latitude of second point in decimal degrees
        lon2: Longitude of second point in decimal degrees
    
    Returns:
        Distance between the two points in meters
    
    Raises:
        ValueError: If any coordinate is invalid
    
    Example:
        >>> # Distance from UB main gate to library
        >>> distance = calculate_distance(5.9631, 10.2588, 5.9645, 10.2595)
        >>> print(f"Distance: {distance:.2f} meters")
        Distance: 173.45 meters
    """
    # Validate all coordinates before calculation
    if not validate_coordinates(lat1, lon1):
        raise ValueError(f"Invalid coordinates for point 1: ({lat1}, {lon1})")
    if not validate_coordinates(lat2, lon2):
        raise ValueError(f"Invalid coordinates for point 2: ({lat2}, {lon2})")
    
    # Convert degrees to radians (required for trigonometric functions)
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    
    # Calculate the differences in latitude and longitude
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    # Apply Haversine formula
    # a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
    a = (math.sin(delta_lat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * 
         math.sin(delta_lon / 2) ** 2)
    
    # c = 2 * atan2(√a, √(1−a))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Distance = radius * c
    distance = EARTH_RADIUS_METERS * c
    
    return distance


def get_cardinal_direction(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    """
    Get the cardinal direction from point 1 to point 2.
    
    Returns one of: N, NE, E, SE, S, SW, W, NW
    
    NOTE: This function is provided for completeness but should NOT be used
    in final user-facing directions. Use natural landmarks instead!
    
    Args:
        lat1: Latitude of starting point in decimal degrees
        lon1: Longitude of starting point in decimal degrees
        lat2: Latitude of destination point in decimal degrees
        lon2: Longitude of destination point in decimal degrees
    
    Returns:
        Cardinal direction as string (N, NE, E, SE, S, SW, W, NW)
    
    Example:
        >>> direction = get_cardinal_direction(5.9631, 10.2588, 5.9645, 10.2595)
        >>> print(f"Direction: {direction}")
        Direction: NE
    """
    # Calculate bearing in degrees
    bearing = calculate_bearing(lat1, lon1, lat2, lon2)
    
    # Define cardinal directions with their angle ranges
    # Each direction covers 45 degrees (360 / 8 directions)
    directions = [
        ("N", 337.5, 22.5),    # North: 337.5° to 22.5°
        ("NE", 22.5, 67.5),    # Northeast: 22.5° to 67.5°
        ("E", 67.5, 112.5),    # East: 67.5° to 112.5°
        ("SE", 112.5, 157.5),  # Southeast: 112.5° to 157.5°
        ("S", 157.5, 202.5),   # South: 157.5° to 202.5°
        ("SW", 202.5, 247.5),  # Southwest: 202.5° to 247.5°
        ("W", 247.5, 292.5),   # West: 247.5° to 292.5°
        ("NW", 292.5, 337.5),  # Northwest: 292.5° to 337.5°
    ]
    
    # Find which direction range the bearing falls into
    for direction, start, end in directions:
        # Special case for North (wraps around 0°)
        if direction == "N":
            if bearing >= start or bearing < end:
                return direction
        else:
            if start <= bearing < end:
                return direction
    
    # Default to North (should never reach here)
    return "N"


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the bearing (direction) from point 1 to point 2.
    
    Bearing is the angle measured clockwise from north (0°-360°).
    
    Args:
        lat1: Latitude of starting point in decimal degrees
        lon1: Longitude of starting point in decimal degrees
        lat2: Latitude of destination point in decimal degrees
        lon2: Longitude of destination point in decimal degrees
    
    Returns:
        Bearing in degrees (0-360, where 0 is North, 90 is East, etc.)
    
    Example:
        >>> bearing = calculate_bearing(5.9631, 10.2588, 5.9645, 10.2595)
        >>> print(f"Bearing: {bearing:.2f}°")
        Bearing: 45.23°
    """
    # Convert coordinates to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon_rad = math.radians(lon2 - lon1)
    
    # Calculate bearing using formula:
    # θ = atan2(sin(Δlong) * cos(lat2), cos(lat1) * sin(lat2) − sin(lat1) * cos(lat2) * cos(Δlong))
    x = math.sin(delta_lon_rad) * math.cos(lat2_rad)
    y = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad))
    
    # Get initial bearing in radians
    bearing_rad = math.atan2(x, y)
    
    # Convert to degrees
    bearing_deg = math.degrees(bearing_rad)
    
    # Normalize to 0-360 range
    # (atan2 returns -180 to +180, we want 0 to 360)
    bearing_normalized = (bearing_deg + 360) % 360
    
    return bearing_normalized


def format_distance(meters: float) -> str:
    """
    Format distance in meters to human-readable string.
    
    Converts to kilometers if distance is >= 1000 meters.
    
    Args:
        meters: Distance in meters
    
    Returns:
        Formatted distance string (e.g., "250 meters" or "1.5 kilometers")
    
    Example:
        >>> print(format_distance(500))
        500 meters
        >>> print(format_distance(1500))
        1.5 kilometers
    """
    # Round to nearest meter
    meters = round(meters)
    
    # If less than 1 kilometer, show in meters
    if meters < 1000:
        return f"{meters} meters"
    
    # Convert to kilometers and format to 1 decimal place
    kilometers = meters / 1000
    
    # If it's a whole number of kilometers, don't show decimal
    if kilometers == int(kilometers):
        return f"{int(kilometers)} kilometers"
    else:
        return f"{kilometers:.1f} kilometers"


def format_time(seconds: int) -> str:
    """
    Format time duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted time string (e.g., "5 minutes" or "1 hour 20 minutes")
    
    Example:
        >>> print(format_time(300))
        5 minutes
        >>> print(format_time(4800))
        1 hour 20 minutes
    """
    # Handle edge case of 0 or negative seconds
    if seconds <= 0:
        return "less than a minute"
    
    # Round to nearest second
    seconds = round(seconds)
    
    # Calculate hours and remaining minutes
    hours = seconds // 3600
    remaining_seconds = seconds % 3600
    minutes = remaining_seconds // 60
    
    # Build the time string
    parts = []
    
    if hours > 0:
        # Format hours (singular or plural)
        if hours == 1:
            parts.append("1 hour")
        else:
            parts.append(f"{hours} hours")
    
    if minutes > 0:
        # Format minutes (singular or plural)
        if minutes == 1:
            parts.append("1 minute")
        else:
            parts.append(f"{minutes} minutes")
    
    # If less than a minute, say so
    if not parts:
        return "less than a minute"
    
    # Join parts with space
    return " ".join(parts)


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate if geographic coordinates are within valid ranges.
    
    Valid ranges:
        - Latitude: -90 to +90 degrees
        - Longitude: -180 to +180 degrees
    
    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
    
    Returns:
        True if coordinates are valid, False otherwise
    
    Example:
        >>> validate_coordinates(5.9631, 10.2588)
        True
        >>> validate_coordinates(95.0, 10.0)  # Invalid latitude
        False
    """
    # Check if latitude is within valid range (-90 to +90)
    if lat < -90 or lat > 90:
        return False
    
    # Check if longitude is within valid range (-180 to +180)
    if lon < -180 or lon > 180:
        return False
    
    # Both coordinates are valid
    return True


def is_on_campus(lat: float, lon: float, campus_bounds: Dict[str, float]) -> bool:
    """
    Check if coordinates are within campus boundaries.
    
    Campus boundaries are defined as a rectangular bounding box with
    north, south, east, and west limits.
    
    Args:
        lat: Latitude to check in decimal degrees
        lon: Longitude to check in decimal degrees
        campus_bounds: Dictionary with keys 'north', 'south', 'east', 'west'
                       containing the campus boundary coordinates
    
    Returns:
        True if coordinates are within campus bounds, False otherwise
    
    Example:
        >>> # Define University of Bamenda campus bounds
        >>> campus = {
        ...     'north': 5.97,
        ...     'south': 5.95,
        ...     'east': 10.16,
        ...     'west': 10.14
        ... }
        >>> is_on_campus(5.9631, 10.2588, campus)
        False  # This point is outside the campus
    """
    # First validate that the coordinates are geographically valid
    if not validate_coordinates(lat, lon):
        return False
    
    # Check if all required boundary keys exist
    required_keys = {'north', 'south', 'east', 'west'}
    if not all(key in campus_bounds for key in required_keys):
        raise ValueError(f"campus_bounds must contain keys: {required_keys}")
    
    # Extract boundary values
    north = campus_bounds['north']
    south = campus_bounds['south']
    east = campus_bounds['east']
    west = campus_bounds['west']
    
    # Validate boundary values themselves
    if not validate_coordinates(north, west):
        raise ValueError(f"Invalid northwest corner: ({north}, {west})")
    if not validate_coordinates(south, east):
        raise ValueError(f"Invalid southeast corner: ({south}, {east})")
    
    # Check if point is within the rectangular bounds
    # Latitude must be between south and north
    lat_in_bounds = south <= lat <= north
    
    # Longitude must be between west and east
    lon_in_bounds = west <= lon <= east
    
    # Point is on campus only if both conditions are true
    return lat_in_bounds and lon_in_bounds


# Usage examples and testing
if __name__ == "__main__":
    # Example 1: Calculate distance between two points on campus
    print("=" * 60)
    print("Example 1: Calculate Distance")
    print("=" * 60)
    
    # Main gate coordinates
    gate_lat, gate_lon = 6.010317, 10.258816
    # Library coordinates (example)
    library_lat, library_lon = 6.010497, 10.259855
    
    distance = calculate_distance(gate_lat, gate_lon, library_lat, library_lon)
    print(f"Distance from main gate to library: {format_distance(distance)}")
    
    # Example 2: Get direction
    print("\n" + "=" * 60)
    print("Example 2: Calculate Direction")
    print("=" * 60)
    
    direction = get_cardinal_direction(gate_lat, gate_lon, library_lat, library_lon)
    bearing = calculate_bearing(gate_lat, gate_lon, library_lat, library_lon)
    print(f"Direction: {direction}")
    print(f"Precise bearing: {bearing:.2f}°")
    
    # Example 3: Format time
    print("\n" + "=" * 60)
    print("Example 3: Format Time")
    print("=" * 60)
    
    print(f"300 seconds = {format_time(300)}")
    print(f"4800 seconds = {format_time(4800)}")
    print(f"45 seconds = {format_time(45)}")
    
    # Example 4: Check if on campus
    print("\n" + "=" * 60)
    print("Example 4: Check Campus Bounds")
    print("=" * 60)
    
    # Define University of Bamenda campus bounds (example)
    ub_campus = {
        'north': 6.012,
        'south': 6.009,
        'east': 10.261,
        'west': 10.257
    }
    
    test_points = [
        (6.010317, 10.258816, "Main gate"),
        (6.010497, 10.259855, "Library"),
        (6.015, 10.270, "Off campus - North"),
        (6.005, 10.250, "Off campus - South")
    ]
    
    for lat, lon, name in test_points:
        on_campus = is_on_campus(lat, lon, ub_campus)
        status = "✅ ON CAMPUS" if on_campus else "❌ OFF CAMPUS"
        print(f"{name}: {status}")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)