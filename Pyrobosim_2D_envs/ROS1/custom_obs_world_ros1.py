#!/usr/bin/env python3
"""
Custom Narrow World for PyRoboSim (ROS 1 Version)
Creates a narrow, winding path with 4 agents at the entrance with camera feeds.
"""

# Use pyrobosim's built-in GUI system
import sys
import os

import numpy as np
try:
    from pyrobosim.core.world import World, Location
except Exception as e:
    print(f"Warning: Could not import World normally: {e}")
    # Try alternative import
    from pyrobosim.core.world import World, Location
from pyrobosim.core.room import Room
from pyrobosim.core.robot import Robot
from pyrobosim.utils.pose import Pose
from shapely.geometry import Polygon, Point
import time
import threading

# Try to import ROS 1 dependencies
ROS_AVAILABLE = False
try:
    import rospy
    from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
    from sensor_msgs.msg import Image, LaserScan, CameraInfo
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Header, String
    try:
        import cv2
        from cv_bridge import CvBridge
        CV_BRIDGE_AVAILABLE = True
    except ImportError:
        CV_BRIDGE_AVAILABLE = False
        print("  Note: cv_bridge not available - camera images will be simplified")
    ROS_AVAILABLE = True
    print("ROS 1 dependencies available - topic publishing enabled")
except ImportError as e:
    import sys
    error_msg = str(e)
    print(f"ROS 1 not available: {error_msg}")
    
    # Provide helpful guidance based on the error
    if 'rospkg' in error_msg or 'rospy' in error_msg or 'geometry_msgs' in error_msg:
        print("\n⚠️  ROS1 packages not found in Python environment.")
        print("   This usually means ROS1 setup files are not sourced.")
        print("\n   Try one of these solutions:")
        print("   1. Source ROS1 Noetic setup.bash before running:")
        print("      source /opt/ros/noetic/setup.bash")
        print("      python3 custom_narrow_world_ros1.py")
        print("\n   2. Or if using conda, install rospkg:")
        print("      conda install -c conda-forge rospkg")
        print("      pip install rospkg")
        print("\n   3. Or source ROS in your ~/.bashrc:")
        print("      echo 'source /opt/ros/noetic/setup.bash' >> ~/.bashrc")
        print("      source ~/.bashrc")
    
    print("\nTopic publishing will be simulated (not actual ROS topics)")
    ROS_AVAILABLE = False
    CV_BRIDGE_AVAILABLE = False
# Only import matplotlib for saving images (if needed)
# PyRoboSim has its own GUI system which we'll use for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Only for saving images if needed
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except:
    MATPLOTLIB_AVAILABLE = False

custom_agents_positions = [
    (1, 2.25),
    (1, 2.75),
    (1, 3.25),
    (1, 3.75)]

def create_narrow_world():
    """Create a custom narrow world matching the image description."""
    
    # Initialize the world
    world = World()
    main_room_coords = [
        (0.0, 0.0), (15.0, 0.0), (15.0, 10.0), (0.0, 10.0)
    ]
    
    # Define the narrow path boundaries based on the image description
    # The path: BROAD at entrance -> narrows down to SMALL PASSAGE -> widens at exit
    # Following the image: starts wide, narrows dramatically in middle-left, narrow passage, widens at end
    
    # Top wall coordinates (from left to right)
    # Entrance is wide (around y=4.5), narrows to passage (y=3.0), widens at exit (y=4.5)
    top_wall_points = [
        (0.0, 4.5),      # Far left entrance - WIDE
        (1.5, 4.5),      # Entrance area - stays wide
        (2.5, 4.0),      # Starts angling inwards
        (3.0, 3.5),      # Narrows more
        (3.5, 3.2),      # Narrows to passage
        (4.0, 3.0),      # NARROW PASSAGE (smallest width ~0.5 units)
        (5.0, 3.0),      # Narrow passage continues
        (6.0, 3.0),      # Narrow passage continues
        (7.0, 3.0),      # Narrow passage continues
        (8.0, 3.0),      # Narrow passage continues
        (9.0, 3.0),      # Narrow passage continues
        (10.0, 3.0),     # Narrow passage continues
        (10.5, 3.2),     # Starts widening
        (11.0, 3.5),     # Widens more
        (11.5, 4.0),     # Widens towards exit
        (12.0, 4.5),     # Far right exit - WIDE
    ]
    
    # Bottom wall coordinates (from left to right)
    # Entrance is wide (around y=1.5), narrows to passage (y=2.5), widens at exit (y=1.5)
    bottom_wall_points = [
        (0.0, 1.5),      # Far left entrance - WIDE
        (1.5, 1.5),      # Entrance area - stays wide
        (2.5, 2.0),      # Starts angling inwards
        (2.6, 2.0),      # Dark block position (smaller gap)
        (2.8, 2.0),      # After dark block
        (3.0, 2.3),      # Narrows more
        (3.5, 2.6),      # Narrows to passage
        (4.0, 2.5),      # NARROW PASSAGE (smallest width ~0.5 units)
        (5.0, 2.5),      # Narrow passage continues
        (6.0, 2.5),      # Narrow passage continues
        (7.0, 2.5),      # Narrow passage continues
        (8.0, 2.5),      # Narrow passage continues
        (9.0, 2.5),      # Narrow passage continues
        (10.0, 2.5),     # Narrow passage continues
        (10.5, 2.6),     # Starts widening
        (11.0, 2.3),     # Widens more
        (11.5, 2.0),     # Widens towards exit
        (12.0, 1.5),     # Far right exit - WIDE
    ]
    
    # Create the walkable area (the path itself)
    # Combine top and bottom walls to create the path polygon
    # Room expects a list of coordinates, not a Polygon object
    path_points = top_wall_points + list(reversed(bottom_wall_points))
    
    # Ensure the polygon is closed (first point == last point)
    if path_points[0] != path_points[-1]:
        path_points.append(path_points[0])
    
    # Create a room for the narrow path
    # Use add_room with keyword arguments
    world.add_room(
        name="narrow_path",
        footprint=main_room_coords,  # Pass coordinates as list
        color=[0.8, 0.8, 0.8]  # Light grey for the path
    )
    
    # Add many obstacles to the main room
    # In pyrobosim, obstacles are represented as rooms with different colors
    obstacles = []
    
    # Rectangular block obstacles scattered throughout
    obstacles.append({
        'name': 'obstacle_1',
        'coords': [(1.0, 1.0), (1.5, 1.0), (1.5, 2.0), (1.0, 2.0)],
        'color': [0.5, 0.5, 0.5]  # Dark gray
    })
    obstacles.append({
        'name': 'obstacle_2',
        'coords': [(2.5, 1.5), (3.0, 1.5), (3.0, 2.5), (2.5, 2.5)],
        'color': [0.5, 0.5, 0.5]
    })
    obstacles.append({
        'name': 'obstacle_3',
        'coords': [(4.5, 1.0), (5.0, 1.0), (5.0, 1.8), (4.5, 1.8)],
        'color': [0.5, 0.5, 0.5]
    })
    obstacles.append({
        'name': 'obstacle_4',
        'coords': [(6.0, 1.2), (6.5, 1.2), (6.5, 2.0), (6.0, 2.0)],
        'color': [0.5, 0.5, 0.5]
    })
    obstacles.append({
        'name': 'obstacle_5',
        'coords': [(7.5, 1.0), (8.0, 1.0), (8.0, 1.8), (7.5, 1.8)],
        'color': [0.5, 0.5, 0.5]
    })
    obstacles.append({
        'name': 'obstacle_6',
        'coords': [(9.0, 1.3), (9.5, 1.3), (9.5, 2.2), (9.0, 2.2)],
        'color': [0.5, 0.5, 0.5]
    })
    obstacles.append({
        'name': 'obstacle_7',
        'coords': [(10.5, 1.0), (11.0, 1.0), (11.0, 1.7), (10.5, 1.7)],
        'color': [0.5, 0.5, 0.5]
    })
    obstacles.append({
        'name': 'obstacle_8',
        'coords': [(12.5, 1.2), (13.0, 1.2), (13.0, 2.0), (12.5, 2.0)],
        'color': [0.5, 0.5, 0.5]
    })
    
    # Obstacles in the upper area
    obstacles.append({
        'name': 'obstacle_9',
        'coords': [(1.5, 4.0), (2.0, 4.0), (2.0, 5.0), (1.5, 5.0)],
        'color': [0.5, 0.5, 0.5]
    })
    obstacles.append({
        'name': 'obstacle_10',
        'coords': [(3.0, 4.5), (3.5, 4.5), (3.5, 5.5), (3.0, 5.5)],
        'color': [0.5, 0.5, 0.5]
    })
    obstacles.append({
        'name': 'obstacle_11',
        'coords': [(5.0, 4.2), (5.5, 4.2), (5.5, 5.2), (5.0, 5.2)],
        'color': [0.5, 0.5, 0.5]
    })
    obstacles.append({
        'name': 'obstacle_12',
        'coords': [(6.5, 4.0), (7.0, 4.0), (7.0, 5.0), (6.5, 5.0)],
        'color': [0.5, 0.5, 0.5]
    })
    obstacles.append({
        'name': 'obstacle_13',
        'coords': [(8.0, 4.3), (8.5, 4.3), (8.5, 5.3), (8.0, 5.3)],
        'color': [0.5, 0.5, 0.5]
    })
    obstacles.append({
        'name': 'obstacle_14',
        'coords': [(9.5, 4.0), (10.0, 4.0), (10.0, 5.0), (9.5, 5.0)],
        'color': [0.5, 0.5, 0.5]
    })
    obstacles.append({
        'name': 'obstacle_15',
        'coords': [(11.0, 4.2), (11.5, 4.2), (11.5, 5.2), (11.0, 5.2)],
        'color': [0.5, 0.5, 0.5]
    })
    obstacles.append({
        'name': 'obstacle_16',
        'coords': [(12.5, 4.5), (13.0, 4.5), (13.0, 5.5), (12.5, 5.5)],
        'color': [0.5, 0.5, 0.5]
    })
    
    # Larger obstacles
    obstacles.append({
        'name': 'obstacle_17',
        'coords': [(2.0, 6.0), (3.0, 6.0), (3.0, 7.0), (2.0, 7.0)],
        'color': [0.4, 0.4, 0.4]  # Darker gray
    })
    obstacles.append({
        'name': 'obstacle_18',
        'coords': [(5.0, 6.5), (6.0, 6.5), (6.0, 7.5), (5.0, 7.5)],
        'color': [0.4, 0.4, 0.4]
    })
    obstacles.append({
        'name': 'obstacle_19',
        'coords': [(8.0, 6.0), (9.0, 6.0), (9.0, 7.0), (8.0, 7.0)],
        'color': [0.4, 0.4, 0.4]
    })
    obstacles.append({
        'name': 'obstacle_20',
        'coords': [(11.0, 6.5), (12.0, 6.5), (12.0, 7.5), (11.0, 7.5)],
        'color': [0.4, 0.4, 0.4]
    })
    
    # L-shaped obstacles
    obstacles.append({
        'name': 'obstacle_l1',
        'coords': [(3.5, 7.5), (4.5, 7.5), (4.5, 8.5), (4.0, 8.5), (4.0, 9.0), (3.5, 9.0)],
        'color': [0.6, 0.6, 0.6]
    })
    obstacles.append({
        'name': 'obstacle_l2',
        'coords': [(9.5, 7.5), (10.5, 7.5), (10.5, 8.5), (10.0, 8.5), (10.0, 9.0), (9.5, 9.0)],
        'color': [0.6, 0.6, 0.6]
    })
    
    # U-shaped obstacles
    obstacles.append({
        'name': 'obstacle_u1',
        'coords': [(1.0, 8.0), (2.0, 8.0), (2.0, 9.0), (1.5, 9.0), (1.5, 8.5), (1.0, 8.5)],
        'color': [0.6, 0.6, 0.6]
    })
    obstacles.append({
        'name': 'obstacle_u2',
        'coords': [(13.0, 8.0), (14.0, 8.0), (14.0, 9.0), (13.5, 9.0), (13.5, 8.5), (13.0, 8.5)],
        'color': [0.6, 0.6, 0.6]
    })
    
    # Add all obstacles as rooms (pyrobosim represents obstacles as rooms)
    # Note: In pyrobosim, rooms are walkable areas, but we can use them as obstacles
    # by making them visually distinct and ensuring they render on top
    from shapely.plotting import patch_from_polygon
    
    print(f"  Creating {len(obstacles)} obstacles...")
    for obs in obstacles:
        obstacle_room = world.add_room(
            name=obs['name'],
            footprint=obs['coords'],
            color=obs['color']
        )
        # Ensure the obstacle room has proper visualization
        if obstacle_room:
            # Override the visualization to show as filled polygon instead of just walls
            # Normal rooms show walls (buffered_polygon - polygon), but obstacles should be filled
            try:
                # Create a filled patch from the room polygon itself (not the wall outline)
                obstacle_room.viz_patch = patch_from_polygon(
                    obstacle_room.polygon,  # Use the room polygon directly, not the wall outline
                    facecolor=obs['color'],
                    edgecolor='black',
                    linewidth=1.5,
                    alpha=0.9,
                    zorder=10  # High z-order to appear on top
                )
                print(f"    Created obstacle: {obs['name']} at {obs['coords'][0]}")
            except Exception as e:
                print(f"    Warning: Could not create filled patch for {obs['name']}: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to normal visualization
                obstacle_room.update_visualization_polygon()
    
    # Force world to update all room visualizations
    # This ensures all rooms (including obstacles) are properly rendered
    for room in world.rooms:
        if hasattr(room, 'update_visualization_polygon'):
            # Only update if viz_patch wasn't already set (for obstacles)
            if not hasattr(room, 'viz_patch') or room.viz_patch is None:
                room.update_visualization_polygon()
    
    print(f"  ✓ Created {len(obstacles)} obstacles (total rooms: {len(world.rooms)})")
    
    # Note: Walls are represented by the room boundaries
    # The path polygon defines the walkable area, and areas outside are obstacles
    
    # Add a location at the exit (right side) marked by green square
    # Use a valid category like "table" or "desk" instead of "target"
    # world.add_location(
    #     name="exit",
    #     category="table",  # Use a valid category
    #     parent=world.rooms[0],
    #     pose=Pose(x=10.5, y=3.0, yaw=0.0)
    # )
    
    return world

def create_obstacle_world():
    """Create a world with simple filled polygon obstacles that block robot movement."""
    from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint
    from shapely.ops import unary_union
    
    world = World()
    
    # Create main room (walkable area)
    main_room_coords = [(0.0, 0.0), (15.0, 0.0), (15.0, 10.0), (0.0, 10.0)]
    main_room = world.add_room(
        name="main_area",
        footprint=main_room_coords,
        color=[0.8, 0.8, 0.8]
    )
    
    # Store obstacle data for rendering AND collision
    world.obstacle_polygons = []
    world.obstacle_shapely_polygons = []  # For collision detection
    
    # Define polygon obstacles as simple coordinates
    polygon_obstacles = [
        # Rectangular blocks in lower area
        {'coords': [(1.0, 1.0), (1.5, 1.0), (1.5, 2.0), (1.0, 2.0)], 'color': [0.5, 0.5, 0.5], 'type': 'polygon'},
        {'coords': [(2.5, 1.5), (3.0, 1.5), (3.0, 2.5), (2.5, 2.5)], 'color': [0.5, 0.5, 0.5], 'type': 'polygon'},
        {'coords': [(4.5, 1.0), (5.0, 1.0), (5.0, 1.8), (4.5, 1.8)], 'color': [0.5, 0.5, 0.5], 'type': 'polygon'},
        {'coords': [(6.0, 1.2), (6.5, 1.2), (6.5, 2.0), (6.0, 2.0)], 'color': [0.5, 0.5, 0.5], 'type': 'polygon'},
        {'coords': [(7.5, 1.0), (8.0, 1.0), (8.0, 1.8), (7.5, 1.8)], 'color': [0.5, 0.5, 0.5], 'type': 'polygon'},
        {'coords': [(9.0, 1.3), (9.5, 1.3), (9.5, 2.2), (9.0, 2.2)], 'color': [0.5, 0.5, 0.5], 'type': 'polygon'},
        {'coords': [(10.5, 1.0), (11.0, 1.0), (11.0, 1.7), (10.5, 1.7)], 'color': [0.5, 0.5, 0.5], 'type': 'polygon'},
        
        # Rectangular blocks in upper area
        {'coords': [(1.5, 8.0), (2.0, 8.0), (2.0, 9.0), (1.5, 9.0)], 'color': [0.5, 0.5, 0.5], 'type': 'polygon'},
        {'coords': [(3.0, 8.5), (3.5, 8.5), (3.5, 9.5), (3.0, 9.5)], 'color': [0.5, 0.5, 0.5], 'type': 'polygon'},
        {'coords': [(5.0, 8.2), (5.5, 8.2), (5.5, 9.2), (5.0, 9.2)], 'color': [0.5, 0.5, 0.5], 'type': 'polygon'},
        {'coords': [(6.5, 8.0), (7.0, 8.0), (7.0, 9.0), (6.5, 9.0)], 'color': [0.5, 0.5, 0.5], 'type': 'polygon'},
        {'coords': [(8.0, 8.3), (8.5, 8.3), (8.5, 9.3), (8.0, 9.3)], 'color': [0.5, 0.5, 0.5], 'type': 'polygon'},
        {'coords': [(9.5, 8.0), (10.0, 8.0), (10.0, 9.0), (9.5, 9.0)], 'color': [0.5, 0.5, 0.5], 'type': 'polygon'},
        {'coords': [(11.0, 8.2), (11.5, 8.2), (11.5, 9.2), (11.0, 9.2)], 'color': [0.5, 0.5, 0.5], 'type': 'polygon'},
        
        # Larger blocks in middle area
        {'coords': [(2.0, 4.5), (3.0, 4.5), (3.0, 5.5), (2.0, 5.5)], 'color': [0.4, 0.4, 0.4], 'type': 'polygon'},
        {'coords': [(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0)], 'color': [0.4, 0.4, 0.4], 'type': 'polygon'},
        {'coords': [(8.0, 4.5), (9.0, 4.5), (9.0, 5.5), (8.0, 5.5)], 'color': [0.4, 0.4, 0.4], 'type': 'polygon'},
        {'coords': [(11.0, 5.0), (12.0, 5.0), (12.0, 6.0), (11.0, 6.0)], 'color': [0.4, 0.4, 0.4], 'type': 'polygon'},
        
        # L-shaped obstacles
        # {'coords': [(3.5, 2.5), (4.5, 2.5), (4.5, 3.5), (4.0, 3.5), (4.0, 4.0), (3.5, 4.0)], 'color': [0.6, 0.6, 0.6], 'type': 'polygon'},
        {'coords': [(10.5, 2.5), (11.5, 2.5), (11.5, 3.5), (11.0, 3.5), (11.0, 4.0), (10.5, 4.0)], 'color': [0.6, 0.6, 0.6], 'type': 'polygon'},
        
        # U-shaped obstacles
        {'coords': [(1.0, 6.5), (2.0, 6.5), (2.0, 7.5), (1.5, 7.5), (1.5, 7.0), (1.0, 7.0)], 'color': [0.6, 0.6, 0.6], 'type': 'polygon'},
        {'coords': [(13.0, 6.5), (14.0, 6.5), (14.0, 7.5), (13.5, 7.5), (13.5, 7.0), (13.0, 7.0)], 'color': [0.6, 0.6, 0.6], 'type': 'polygon'},
    ]
    
    # Define circular obstacles (center x, y, radius)
    circle_obstacles = [
        {'center': (12.5, 2.0), 'radius': 0.3, 'color': [0.3, 0.3, 0.7], 'type': 'circle'},
        {'center': (13.5, 3.5), 'radius': 0.25, 'color': [0.3, 0.3, 0.7], 'type': 'circle'},
        {'center': (4.0, 7.0), 'radius': 0.35, 'color': [0.3, 0.3, 0.7], 'type': 'circle'},
        {'center': (7.5, 3.0), 'radius': 0.3, 'color': [0.3, 0.3, 0.7], 'type': 'circle'},
        {'center': (6.0, 6.5), 'radius': 0.25, 'color': [0.3, 0.3, 0.7], 'type': 'circle'},
        {'center': (12.0, 7.5), 'radius': 0.35, 'color': [0.3, 0.3, 0.7], 'type': 'circle'},
    ]
    
    # Add all obstacles to the lists
    all_obstacles = polygon_obstacles + circle_obstacles
    world.obstacle_polygons = all_obstacles
    
    # Create Shapely polygons for collision detection
    collision_polygons = []
    for obs in polygon_obstacles:
        shapely_poly = ShapelyPolygon(obs['coords'])
        collision_polygons.append(shapely_poly)
        world.obstacle_shapely_polygons.append(shapely_poly)
    
    # Create Shapely polygons for circles (buffered points)
    for obs in circle_obstacles:
        center_point = ShapelyPoint(obs['center'])
        circle_poly = center_point.buffer(obs['radius'])
        collision_polygons.append(circle_poly)
        world.obstacle_shapely_polygons.append(circle_poly)
    
    # Combine all obstacle polygons into one collision polygon
    # and subtract from the main room's internal collision polygon
    if collision_polygons:
        combined_obstacles = unary_union(collision_polygons)
        world.combined_obstacle_polygon = combined_obstacles  # Store for collision checking
        
        # Update the main room's internal collision polygon to exclude obstacles
        if hasattr(main_room, 'internal_collision_polygon'):
            main_room.internal_collision_polygon = main_room.internal_collision_polygon.difference(combined_obstacles)
            print(f"  ✓ Updated collision geometry to block {len(all_obstacles)} obstacles")
        
        # Also update the collision polygons for the room
        if hasattr(main_room, 'update_collision_polygons'):
            main_room.update_collision_polygons(world.inflation_radius if hasattr(world, 'inflation_radius') else 0.0)
    
    print(f"  Created world with {len(polygon_obstacles)} polygon obstacles and {len(circle_obstacles)} circle obstacles")
    print(f"  Obstacles will block robot movement (collision detection enabled)")
    
    return world


def add_agents_with_cameras(world, num_agents=4, positions=custom_agents_positions):
    """Add multiple agents at the entrance with camera feeds."""
    
    # Entrance position (left side, wider area)
    entrance_x = 0.75
    entrance_y_base = 3.0  # Center of the path at entrance (between 1.5 and 4.5)
    
    agents = []
    
    # Spacing between agents
    agent_spacing = 0.5
    agent_radius = 0.15
    
    for i in range(num_agents):
        agent_name = f"tb_{i}"
        
        # Arrange agents in a 2x2 grid at the entrance
        if i < 2:
            # First row (lower)
            x = entrance_x + i * agent_spacing
            y = entrance_y_base - 0.3
        else:
            # Second row (upper)
            x = entrance_x + (i - 2) * agent_spacing
            y = entrance_y_base + 0.3
        if positions is not None:
            pos = positions[i]
            if len(pos) == 2:
                x, y = float(pos[0]), float(pos[1])
                yaw = 0.0
            elif len(pos) == 3:
                x, y, yaw = float(pos[0]), float(pos[1]), float(pos[2])
            else:
                raise ValueError("Each position must be a tuple of (x, y) or (x, y, yaw)")
        else:
            if i <2:
                x = entrance_x + i * agent_spacing
                y = entrance_y_base - 0.3
            else:
                x = entrance_x + (i - 2) * agent_spacing
                y = entrance_y_base + 0.3
            yaw = 0.0
        
        pose = Pose(x=x, y=y, yaw=0.0)  # Facing right (into the path)
        pose = Pose(x=x, y=y, yaw=yaw)
        
        # Create robot with camera sensor
        # Note: PyRoboSim doesn't have built-in camera sensors, but we'll visualize
        # camera field of view in the visualization
        robot = Robot(
            name=agent_name,
            radius=agent_radius,
            color=[0.2, 0.6, 0.9]  # Blue color for agents
        )
        
        # Store camera info for visualization
        robot.camera_active = True
        robot.camera_fov = np.pi / 4  # 45 degrees
        robot.camera_range = 2.0
        
        world.add_robot(robot, pose=pose)
        # Ensure the pose is set correctly (sometimes add_robot resets it)
        if hasattr(robot, 'pose'):
            robot.pose = pose
        agents.append(robot)
    
    # Verify all robots are added
    print(f"  Added {len(agents)} robots to world (world has {len(world.robots)} robots)")
    for i, robot in enumerate(agents):
        print(f"    Robot {i+1}: {robot.name} at ({robot.pose.x:.2f}, {robot.pose.y:.2f})")
    
    return agents


def setup_ros_topics(world, agents):
    """Set up ROS 1 topic publishers for each robot.
    
    Each robot publishes:
    - /{robot_name}/cmd_vel (Twist) - velocity commands
    - /{robot_name}/odom (Odometry) - odometry
    - /{robot_name}/pose (PoseStamped) - current pose
    - /{robot_name}/camera/image_raw (Image) - camera feed
    - /{robot_name}/camera/camera_info (CameraInfo) - camera info
    
    Returns:
        node_name: ROS 1 node name if available, None otherwise
        publishers: Dictionary of publishers for each robot
    """
    publishers = {}
    
    if not ROS_AVAILABLE:
        print("  Note: ROS 1 not available - topics will be simulated")
        # Create simulated publishers (dummy for non-ROS environments)
        velocity_commands = {}
        for agent in agents:
            publishers[agent.name] = {
                'cmd_vel': f'/{agent.name}/cmd_vel',
                'odom': f'/{agent.name}/odom',
                'pose': f'/{agent.name}/pose',
                'camera_image': f'/{agent.name}/camera/image_raw',
                'camera_info': f'/{agent.name}/camera/camera_info',
            }
            velocity_commands[agent.name] = {
                'linear_x': 0.0,
                'linear_y': 0.0,
                'angular_z': 0.0,
                'timestamp': time.time()
            }
            print(f"  {agent.name} topics:")
            for topic_name, topic_path in publishers[agent.name].items():
                print(f"    - {topic_path} ({topic_name})")
        return None, publishers, velocity_commands
    
    # Initialize ROS 1 node directly (for local setup, just try to initialize)
    # If roscore isn't running, this will fail gracefully
    node_name = 'pyrobosim_world'
    node_initialized = False
    node_fully_initialized = False
    
    try:
        # Check if node is already initialized
        try:
            existing_name = rospy.get_name()
            node_name = existing_name
            print(f"Found existing ROS node: {node_name}")
            
            # CRITICAL: Check if node is FULLY initialized (rospy.Time must work)
            try:
                test_time = rospy.Time.now()
                node_fully_initialized = True
                node_initialized = True
                print(f"✓ ROS node is fully initialized (Time works: {test_time.secs}.{test_time.nsecs})")
            except Exception as time_err:
                # Node exists but not fully initialized - might be timing issue with roscore
                print(f"⚠️  Node exists but rospy.Time not initialized: {time_err}")
                print("   Waiting a moment to see if roscore becomes ready...")
                time.sleep(1.0)  # Wait 1 second for roscore to be ready
                
                # Check again if rospy.Time works now
                try:
                    test_time = rospy.Time.now()
                    node_fully_initialized = True
                    node_initialized = True
                    print(f"✓ rospy.Time now works (Time: {test_time.secs}.{test_time.nsecs})")
                except Exception as time_err2:
                    print(f"   Still not working: {time_err2}")
                    print("   Will try to re-initialize node...")
                    node_initialized = False
                    node_fully_initialized = False
                
        except rospy.ROSException:
            # Node not initialized, initialize it
            node_initialized = False
            node_fully_initialized = False
    
    except Exception as e:
        print(f"Warning: Error checking ROS node: {e}")
        node_initialized = False
        node_fully_initialized = False
    
    # Initialize node if not fully initialized
    if not node_fully_initialized:
        try:
            # Use a clean node name (remove any leading slashes)
            clean_node_name = node_name.lstrip('/').replace('/', '_')
            if clean_node_name == '' or clean_node_name.startswith('unnamed'):
                clean_node_name = 'pyrobosim_world'
            
            # Try to initialize the node with a clean name
            # Use anonymous=True to avoid conflicts if node already exists
            rospy.init_node(clean_node_name, anonymous=True)
            node_initialized = True
            
            # Verify rospy.Time is now working
            try:
                test_time = rospy.Time.now()
                node_fully_initialized = True
                actual_node_name = rospy.get_name()
                print(f"✓ ROS 1 node '{actual_node_name}' initialized successfully")
                print(f"✓ rospy.Time initialized (current time: {test_time.secs}.{test_time.nsecs})")
            except Exception as time_err:
                print(f"✗ ERROR: rospy.Time still not initialized after init_node: {time_err}")
                print("   This usually means roscore is not running properly")
                node_fully_initialized = False
                
        except rospy.exceptions.ROSException as init_err:
            # Node might already be initialized by another process
            # Check if it's actually working now (maybe pyrobosim fixed it)
            try:
                test_time = rospy.Time.now()
                node_fully_initialized = True
                node_initialized = True
                actual_node_name = rospy.get_name()
                print(f"✓ Node already initialized ('{actual_node_name}'), but Time works now")
            except Exception as time_err:
                print(f"✗ Could not initialize ROS node: {init_err}")
                print(f"   And rospy.Time still doesn't work: {time_err}")
                print("   This usually means:")
                print("     1. roscore is not running, OR")
                print("     2. Another process initialized the node before roscore was ready")
                print("   Solution: Make sure roscore is running BEFORE starting this script")
                print("   Falling back to simulation mode")
                node_fully_initialized = False
                node_initialized = False
        except Exception as init_error:
            print(f"✗ Unexpected error initializing ROS node: {init_error}")
            print("   Falling back to simulation mode")
            node_fully_initialized = False
            node_initialized = False
    
    # If node is not fully initialized, fall back to simulation mode
    if not node_fully_initialized:
        print("\n⚠️  ROS node not fully initialized - falling back to simulation mode")
        velocity_commands = {}
        for agent in agents:
            publishers[agent.name] = {
                'cmd_vel': f'/{agent.name}/cmd_vel',
                'odom': f'/{agent.name}/odom',
                'pose': f'/{agent.name}/pose',
                'camera_image': f'/{agent.name}/camera/image_raw',
                'camera_info': f'/{agent.name}/camera/camera_info',
            }
            velocity_commands[agent.name] = {
                'linear_x': 0.0,
                'linear_y': 0.0,
                'angular_z': 0.0,
                'timestamp': time.time()
            }
        return None, publishers, velocity_commands
    
    # Create publishers for each robot
    bridge = None
    if CV_BRIDGE_AVAILABLE:
        try:
            bridge = CvBridge()
        except:
            pass
    
    # Store velocity commands for each robot
    velocity_commands = {}
    
    def cmd_vel_callback(msg, robot_name):
        """Callback for cmd_vel commands"""
        velocity_commands[robot_name] = {
            'linear_x': msg.linear.x,
            'linear_y': msg.linear.y,
            'angular_z': msg.angular.z,
            'timestamp': time.time()
        }
        print(f"[DEBUG] Received cmd_vel for {robot_name}: linear_x={msg.linear.x:.2f}, angular_z={msg.angular.z:.2f}")
    
    for agent in agents:
        robot_name = agent.name
        publishers[robot_name] = {}
        velocity_commands[robot_name] = {
            'linear_x': 0.0,
            'linear_y': 0.0,
            'angular_z': 0.0,
            'timestamp': time.time()
        }
        
        # Velocity command subscriber (for receiving commands)
        cmd_vel_topic = f'/{robot_name}/cmd_vel'
        publishers[robot_name]['cmd_vel_sub'] = rospy.Subscriber(
            cmd_vel_topic, Twist,
            lambda msg, rn=robot_name: cmd_vel_callback(msg, rn),
            queue_size=10
        )
        
        # Odometry publisher
        odom_topic = f'/{robot_name}/odom'
        publishers[robot_name]['odom'] = rospy.Publisher(
            odom_topic, Odometry, queue_size=10
        )
        
        # Pose publisher
        pose_topic = f'/{robot_name}/pose'
        publishers[robot_name]['pose'] = rospy.Publisher(
            pose_topic, PoseStamped, queue_size=10
        )
        
        # Camera image publisher
        camera_image_topic = f'/{robot_name}/camera/image_raw'
        publishers[robot_name]['camera_image'] = rospy.Publisher(
            camera_image_topic, Image, queue_size=10
        )
        
        # Camera info publisher
        camera_info_topic = f'/{robot_name}/camera/camera_info'
        publishers[robot_name]['camera_info'] = rospy.Publisher(
            camera_info_topic, CameraInfo, queue_size=10
        )
        
        # Store bridge for image conversion
        publishers[robot_name]['bridge'] = bridge
        
        print(f"  {robot_name} topics:")
        print(f"    - {odom_topic} (Odometry)")
        print(f"    - {pose_topic} (PoseStamped)")
        print(f"    - {camera_image_topic} (Image)")
        print(f"    - {camera_info_topic} (CameraInfo)")
        print(f"    - {cmd_vel_topic} (Twist - subscribe for commands)")
        
        # Publish initial messages to force topic advertisement (ROS1 requirement)
        # This makes topics visible to 'rostopic list' immediately
        try:
            # Get current time - verify rospy.Time is working
            try:
                current_time = rospy.Time.now()
                print(f"    [DEBUG] Publishing for {robot_name} at time: {current_time.secs}.{current_time.nsecs}")
            except Exception as time_err:
                print(f"    [WARNING] rospy.Time not available for {robot_name}: {time_err}")
                # Try zero time as fallback
                current_time = rospy.Time(0, 0)
                print(f"    [DEBUG] Using zero time for {robot_name}")
            
            # Publish initial odometry message
            odom_msg = Odometry()
            odom_msg.header = Header()
            odom_msg.header.stamp = current_time
            odom_msg.header.frame_id = 'odom'
            odom_msg.child_frame_id = f'{robot_name}/base_link'
            odom_msg.pose.pose.position.x = float(agent.pose.x)
            odom_msg.pose.pose.position.y = float(agent.pose.y)
            odom_msg.pose.pose.position.z = float(agent.pose.z)
            odom_msg.pose.pose.orientation.w = float(agent.pose.q[0])
            odom_msg.pose.pose.orientation.x = float(agent.pose.q[1])
            odom_msg.pose.pose.orientation.y = float(agent.pose.q[2])
            odom_msg.pose.pose.orientation.z = float(agent.pose.q[3])
            publishers[robot_name]['odom'].publish(odom_msg)
            
            # Publish initial pose message
            pose_msg = PoseStamped()
            pose_msg.header = Header()
            pose_msg.header.stamp = current_time
            pose_msg.header.frame_id = 'map'
            pose_msg.pose.position.x = float(agent.pose.x)
            pose_msg.pose.position.y = float(agent.pose.y)
            pose_msg.pose.position.z = float(agent.pose.z)
            pose_msg.pose.orientation.w = float(agent.pose.q[0])
            pose_msg.pose.orientation.x = float(agent.pose.q[1])
            pose_msg.pose.orientation.y = float(agent.pose.q[2])
            pose_msg.pose.orientation.z = float(agent.pose.q[3])
            publishers[robot_name]['pose'].publish(pose_msg)
            
            # Publish initial camera info
            cam_info = CameraInfo()
            cam_info.header = Header()
            cam_info.header.stamp = current_time
            cam_info.header.frame_id = f'{robot_name}/camera_frame'
            cam_info.width = 640
            cam_info.height = 480
            cam_info.distortion_model = 'plumb_bob'
            cam_info.K = [320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0]
            publishers[robot_name]['camera_info'].publish(cam_info)
            
            # Publish initial camera image (dummy)
            if hasattr(agent, 'camera_active') and agent.camera_active:
                img = np.zeros((480, 640, 3), dtype=np.uint8)
                bridge = publishers[robot_name].get('bridge')
                if bridge is not None:
                    try:
                        img_msg = bridge.cv2_to_imgmsg(img, encoding='rgb8')
                        img_msg.header = Header()
                        img_msg.header.stamp = current_time
                        img_msg.header.frame_id = f'{robot_name}/camera_frame'
                        publishers[robot_name]['camera_image'].publish(img_msg)
                    except:
                        img_msg = Image()
                        img_msg.header = Header()
                        img_msg.header.stamp = current_time
                        img_msg.header.frame_id = f'{robot_name}/camera_frame'
                        img_msg.height = 480
                        img_msg.width = 640
                        img_msg.encoding = 'rgb8'
                        img_msg.is_bigendian = 0
                        img_msg.step = 640 * 3
                        img_msg.data = img.tobytes()
                        publishers[robot_name]['camera_image'].publish(img_msg)
            
            # Verify publisher is working
            try:
                num_subscribers = publishers[robot_name]['odom'].get_num_connections()
                print(f"    [DEBUG] {robot_name}/odom publisher created, subscribers: {num_subscribers}")
            except Exception as pub_err:
                print(f"    [DEBUG] Could not verify {robot_name}/odom publisher: {pub_err}")
                
        except Exception as e:
            print(f"  ⚠️  Warning: Could not publish initial messages for {robot_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Process ROS callbacks to register topics with master
    # In ROS1, topics only register when the node processes callbacks
    try:
        # Small spin to process callbacks and register topics
        for _ in range(10):  # Process callbacks 10 times
            rospy.rostime.wallsleep(0.01)  # 10ms sleep
    except Exception as e:
        print(f"  Warning: Could not process callbacks: {e}")
    
    # Give ROS1 time to register topics with master
    time.sleep(0.5)
    
    # Verify publishers are created
    print(f"\n✓ Created {len(agents)} robot publishers")
    print(f"✓ Node name: {node_name}")
    
    # Verify roscore is running and connected
    try:
        master = rospy.get_master()
        if master:
            try:
                # Try to get system state from master
                system_state = master.getSystemState()
                import os
                master_uri = os.environ.get('ROS_MASTER_URI', 'http://localhost:11311')
                print(f"✓ ROS 1 master: Connected and responding")
                print(f"  Master URI: {master_uri}")
            except Exception as master_err:
                print(f"⚠️  ROS 1 master: Connected but error getting state: {master_err}")
        else:
            print(f"✗ ROS 1 master: NOT OK (master is None)")
    except Exception as master_err:
        print(f"✗ ROS 1 master: NOT OK - {master_err}")
        print("  Make sure roscore is running!")
        print("  Run 'roscore' in another terminal, then try again.")
    
    # Note: Topics will be fully registered once the spinner starts
    # We'll verify again after spinner starts
    
    return node_name, publishers, velocity_commands


def update_robot_poses(world, agents, velocity_commands, dt=0.1):
    """Update robot poses based on velocity commands.
    
    Args:
        world: World object
        agents: List of robot agents
        velocity_commands: Dictionary of velocity commands for each robot
        dt: Time step in seconds
    """
    for agent in agents:
        if agent.name not in velocity_commands:
            continue
        
        cmd = velocity_commands[agent.name]
        linear_x = cmd['linear_x']
        angular_z = cmd['angular_z']
        
        # Skip if no movement
        if abs(linear_x) < 0.001 and abs(angular_z) < 0.001:
            continue
        
        # Get current pose (store original for collision detection)
        original_x = agent.pose.x
        original_y = agent.pose.y
        original_yaw = agent.pose.get_yaw()
        
        x = original_x
        y = original_y
        yaw = original_yaw
        
        # Calculate dtheta (always needed, even if 0 for straight motion)
        dtheta = angular_z * dt
        
        # Update pose based on velocity
        # For differential drive robots
        if abs(angular_z) < 0.001:
            # Straight motion
            x += linear_x * np.cos(yaw) * dt
            y += linear_x * np.sin(yaw) * dt
        else:
            # Arc motion
            radius = linear_x / angular_z if abs(angular_z) > 0.001 else 0
            x += radius * (np.sin(yaw + dtheta) - np.sin(yaw))
            y += radius * (-np.cos(yaw + dtheta) + np.cos(yaw))
            yaw += dtheta
        
        # Normalize yaw to [-pi, pi]
        yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
        
        # Check collision with walls and obstacles BEFORE updating pose
        collision = False
        if hasattr(world, 'rooms') and world.rooms:
            room = world.rooms[0]
            if hasattr(room, 'polygon') and room.polygon:
                from shapely.geometry import Point
                # Create a buffered point representing the robot's footprint
                robot_radius = agent.radius if hasattr(agent, 'radius') else 0.15
                robot_point = Point(x, y)
                robot_footprint = robot_point.buffer(robot_radius)
                
                # Check if robot is outside the room boundary
                if not room.polygon.contains(robot_point):
                    collision = True
                
                # Check collision with obstacles directly
                if not collision and hasattr(world, 'combined_obstacle_polygon'):
                    # Check if robot footprint intersects any obstacle
                    if robot_footprint.intersects(world.combined_obstacle_polygon):
                        collision = True
                
                # Also check using internal_collision_polygon (backup)
                if not collision and hasattr(room, 'internal_collision_polygon'):
                    if not room.internal_collision_polygon.contains(robot_point):
                        collision = True
        
        # Update pose - use direct assignment which works reliably
        if collision:
            # Revert to original position if collision (don't update pose)
            agent.pose.x = original_x
            agent.pose.y = original_y
            agent.pose.set_euler_angles(0, 0, original_yaw)
        else:
            # Update to new position
            agent.pose.x = x
            agent.pose.y = y
            agent.pose.set_euler_angles(0, 0, yaw)
        
        # Debug output (only print occasionally to avoid spam)
        if int(time.time() * 10) % 10 == 0:  # Print roughly once per second
            if collision:
                print(f"[DEBUG] {agent.name} collision detected, pose reverted")
            else:
                print(f"[DEBUG] Updated {agent.name} pose: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")


def publish_robot_data(node_name, publishers, world, agents):
    """Publish data for all robots to their ROS topics."""
    if not ROS_AVAILABLE or node_name is None:
        return
    
    # Get current time - handle case where rospy.Time might not be initialized
    try:
        current_time = rospy.Time.now()
    except Exception as e:
        print(f"[DEBUG] rospy.Time not available: {e}, using zero time")
        current_time = rospy.Time(0, 0)
    
    for agent in agents:
        if agent.name not in publishers:
            continue
        
        pub_dict = publishers[agent.name]
        
        # Publish odometry
        if 'odom' in pub_dict:
            odom_msg = Odometry()
            odom_msg.header = Header()
            odom_msg.header.stamp = current_time
            odom_msg.header.frame_id = 'odom'
            odom_msg.child_frame_id = f'{agent.name}/base_link'
            odom_msg.pose.pose.position.x = float(agent.pose.x)
            odom_msg.pose.pose.position.y = float(agent.pose.y)
            odom_msg.pose.pose.position.z = float(agent.pose.z)
            odom_msg.pose.pose.orientation.w = float(agent.pose.q[0])
            odom_msg.pose.pose.orientation.x = float(agent.pose.q[1])
            odom_msg.pose.pose.orientation.y = float(agent.pose.q[2])
            odom_msg.pose.pose.orientation.z = float(agent.pose.q[3])
            pub_dict['odom'].publish(odom_msg)
        
        # Publish pose
        if 'pose' in pub_dict:
            pose_msg = PoseStamped()
            pose_msg.header = Header()
            pose_msg.header.stamp = current_time
            pose_msg.header.frame_id = 'map'
            pose_msg.pose.position.x = float(agent.pose.x)
            pose_msg.pose.position.y = float(agent.pose.y)
            pose_msg.pose.position.z = float(agent.pose.z)
            pose_msg.pose.orientation.w = float(agent.pose.q[0])
            pose_msg.pose.orientation.x = float(agent.pose.q[1])
            pose_msg.pose.orientation.y = float(agent.pose.q[2])
            pose_msg.pose.orientation.z = float(agent.pose.q[3])
            pub_dict['pose'].publish(pose_msg)
        
        # Publish camera image (simulated - using a simple image)
        if 'camera_image' in pub_dict and hasattr(agent, 'camera_active') and agent.camera_active:
            # Create a simple simulated camera image (640x480)
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add some visual content (simulated camera view)
            if CV_BRIDGE_AVAILABLE and 'cv2' in globals():
                try:
                    cv2.rectangle(img, (100, 100), (540, 380), (100, 100, 100), -1)
                    cv2.putText(img, f'{agent.name} Camera', (200, 250), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                except:
                    pass
            
            bridge = pub_dict.get('bridge')
            if bridge is not None:
                try:
                    img_msg = bridge.cv2_to_imgmsg(img, encoding='rgb8')
                    img_msg.header = Header()
                    img_msg.header.stamp = current_time
                    img_msg.header.frame_id = f'{agent.name}/camera_frame'
                    pub_dict['camera_image'].publish(img_msg)
                except Exception as e:
                    # If cv_bridge fails, create a simple Image message
                    img_msg = Image()
                    img_msg.header = Header()
                    img_msg.header.stamp = current_time
                    img_msg.header.frame_id = f'{agent.name}/camera_frame'
                    img_msg.height = 480
                    img_msg.width = 640
                    img_msg.encoding = 'rgb8'
                    img_msg.is_bigendian = 0
                    img_msg.step = 640 * 3
                    img_msg.data = img.tobytes()
                    pub_dict['camera_image'].publish(img_msg)
            
            # Publish camera info
            if 'camera_info' in pub_dict:
                cam_info = CameraInfo()
                cam_info.header = Header()
                cam_info.header.stamp = current_time
                cam_info.header.frame_id = f'{agent.name}/camera_frame'
                cam_info.width = 640
                cam_info.height = 480
                cam_info.distortion_model = 'plumb_bob'
                # Simple camera matrix (focal length, principal point)
                cam_info.K = [320.0, 0.0, 320.0,
                             0.0, 320.0, 240.0,
                             0.0, 0.0, 1.0]
                pub_dict['camera_info'].publish(cam_info)
    
    # Process callbacks (non-blocking)
    rospy.rostime.wallsleep(0.001)


def add_custom_elements_to_plot(ax, world, agents):
    """Add custom elements (grid, camera FOV) to pyrobosim's plot."""
    # Add grid overlay on the right half
    grid_start_x = 4.8
    grid_end_x = 12.0
    grid_start_y = 0.0
    grid_end_y = 6.0
    grid_spacing = 0.5
    
    x_grid = np.arange(grid_start_x, grid_end_x + grid_spacing, grid_spacing)
    y_grid = np.arange(grid_start_y, grid_end_y + grid_spacing, grid_spacing)
    
    for x in x_grid:
        ax.axvline(x, color='lightgrey', linewidth=0.5, alpha=0.5)
    for y in y_grid:
        ax.axhline(y, color='lightgrey', linewidth=0.5, alpha=0.5)
    
    # Add camera FOV indicators for each robot
    robots_to_plot = world.robots if hasattr(world, 'robots') and len(world.robots) > 0 else agents
    for agent in robots_to_plot:
        if hasattr(agent, 'pose') and agent.pose and hasattr(agent, 'camera_active') and agent.camera_active:
            yaw = agent.pose.get_yaw()
            fov_angle = getattr(agent, 'camera_fov', np.pi / 4)
            fov_length = getattr(agent, 'camera_range', 2.0)
            
            # Draw FOV cone
            for angle_offset in [-fov_angle/2, fov_angle/2]:
                angle = yaw + angle_offset
                end_x = agent.pose.x + fov_length * np.cos(angle)
                end_y = agent.pose.y + fov_length * np.sin(angle)
                ax.plot(
                    [agent.pose.x, end_x],
                    [agent.pose.y, end_y],
                    'r--', linewidth=1.5, alpha=0.6, zorder=9
                )


def visualize_world_with_grid(world, agents):
    """Visualize the world with grid overlay and camera feeds."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Draw background
    ax.set_facecolor('lightgray')
    
    # Draw all rooms (main room first, then obstacles)
    if world.rooms:
        # Draw main room (first room)
        main_room = world.rooms[0]
        if hasattr(main_room, 'polygon') and main_room.polygon:
            # Get the polygon coordinates
            x, y = main_room.polygon.exterior.xy
            # Draw the main room (walkable area) in light gray
            room_color = main_room.color if hasattr(main_room, 'color') else [0.8, 0.8, 0.8]
            ax.fill(x, y, color=room_color, alpha=0.9, edgecolor='darkgray', linewidth=2, zorder=2)
            # Also draw the outline
            ax.plot(x, y, color='darkgray', linewidth=2, zorder=3, label='Main room')
        else:
            # Fallback: draw from the original coordinates if polygon not available
            top_wall_points = [
                (0.0, 4.5), (1.5, 4.5), (2.5, 4.0), (3.0, 3.5), (3.5, 3.2), (4.0, 3.0),
                (5.0, 3.0), (6.0, 3.0), (7.0, 3.0), (8.0, 3.0), (9.0, 3.0), (10.0, 3.0),
                (10.5, 3.2), (11.0, 3.5), (11.5, 4.0), (12.0, 4.5)
            ]
            bottom_wall_points = [
                (0.0, 1.5), (1.5, 1.5), (2.5, 2.0), (2.6, 2.0), (2.8, 2.0), (3.0, 2.3),
                (3.5, 2.6), (4.0, 2.5), (5.0, 2.5), (6.0, 2.5), (7.0, 2.5), (8.0, 2.5),
                (9.0, 2.5), (10.0, 2.5), (10.5, 2.6), (11.0, 2.3), (11.5, 2.0), (12.0, 1.5)
            ]
            path_points = top_wall_points + list(reversed(bottom_wall_points)) + [top_wall_points[0]]
            path_x = [p[0] for p in path_points]
            path_y = [p[1] for p in path_points]
            ax.fill(path_x, path_y, color='white', alpha=0.9, edgecolor='darkgray', linewidth=3, zorder=2)
            ax.plot(path_x, path_y, color='darkgray', linewidth=3, zorder=3)
        
        # Draw all obstacles (all rooms after the first one)
        for room in world.rooms[1:]:
            if hasattr(room, 'polygon') and room.polygon:
                x, y = room.polygon.exterior.xy
                # Use room color if available, otherwise use dark gray
                obstacle_color = room.color if hasattr(room, 'color') else [0.5, 0.5, 0.5]
                ax.fill(x, y, color=obstacle_color, alpha=0.9, edgecolor='black', linewidth=1.5, zorder=4)
                ax.plot(x, y, color='black', linewidth=1.5, zorder=5)
    
    # Draw walls based on the actual path boundaries
    # Top wall area (above the path)
    # We'll draw a wall that extends from the top boundary upward
    top_wall_height = 1.5
    top_wall_polygon = Polygon([
        (0.0, 6.0), (12.0, 6.0),  # Top of screen
        (12.0, 4.5), (11.5, 4.0), (11.0, 3.5), (10.5, 3.2),  # Following path boundary
        (10.0, 3.0), (9.0, 3.0), (8.0, 3.0), (7.0, 3.0), (6.0, 3.0), (5.0, 3.0), (4.0, 3.0),
        (3.5, 3.2), (3.0, 3.5), (2.5, 4.0), (1.5, 4.5), (0.0, 4.5)  # Back to start
    ])
    top_wall_x, top_wall_y = top_wall_polygon.exterior.xy
    ax.fill(top_wall_x, top_wall_y, color='gray', alpha=0.8, edgecolor='darkgray', linewidth=2, zorder=1)
    
    # Bottom wall area (below the path)
    bottom_wall_polygon = Polygon([
        (0.0, 0.0), (1.5, 1.5), (2.5, 2.0), (2.6, 2.0), (2.8, 2.0), (3.0, 2.3), (3.5, 2.6),
        (4.0, 2.5), (5.0, 2.5), (6.0, 2.5), (7.0, 2.5), (8.0, 2.5), (9.0, 2.5), (10.0, 2.5),
        (10.5, 2.6), (11.0, 2.3), (11.5, 2.0), (12.0, 1.5), (12.0, 0.0), (0.0, 0.0)  # Bottom of screen
    ])
    bottom_wall_x, bottom_wall_y = bottom_wall_polygon.exterior.xy
    ax.fill(bottom_wall_x, bottom_wall_y, color='gray', alpha=0.8, edgecolor='darkgray', linewidth=2, zorder=1)
    
    # Draw dark block on bottom wall (around x=2.6, embedded in wall)
    dark_block = Rectangle((2.6, 1.8), 0.2, 0.3,
                          facecolor='darkgray', edgecolor='black', linewidth=1, zorder=4)
    ax.add_patch(dark_block)
    
    # Add grid overlay on the right half (as per image description)
    # Grid covers approximately 60% of the width (right half)
    grid_start_x = 4.8  # Start at ~40% of 12 units
    grid_end_x = 12.0
    grid_start_y = 0.0
    grid_end_y = 6.0
    grid_spacing = 0.5
    
    # Draw grid lines
    x_grid = np.arange(grid_start_x, grid_end_x + grid_spacing, grid_spacing)
    y_grid = np.arange(grid_start_y, grid_end_y + grid_spacing, grid_spacing)
    
    for x in x_grid:
        ax.axvline(x, color='lightgrey', linewidth=0.5, alpha=0.5)
    for y in y_grid:
        ax.axhline(y, color='lightgrey', linewidth=0.5, alpha=0.5)
    
    # Add green square at exit location (matching image)
    if world.locations:
        exit_location = world.locations[0]
        green_square = Rectangle(
            (exit_location.pose.x - 0.25, exit_location.pose.y - 0.25),
            0.5, 0.5,
            facecolor='green',
            edgecolor='darkgreen',
            linewidth=2,
            alpha=0.7
        )
        ax.add_patch(green_square)
    
    # Plot agents with camera indicators
    # Use world.robots if available, otherwise use agents list
    robots_to_plot = world.robots if hasattr(world, 'robots') and len(world.robots) > 0 else agents
    print(f"  Plotting {len(robots_to_plot)} robots/agents")
    
    for agent in robots_to_plot:
        if hasattr(agent, 'pose') and agent.pose:
            # Draw agent circle
            circle = Circle(
                (agent.pose.x, agent.pose.y),
                agent.radius,
                color=agent.color if hasattr(agent, 'color') else 'blue',
                alpha=0.8,
                zorder=10
            )
            ax.add_patch(circle)
            
            # Get yaw angle from quaternion
            yaw = agent.pose.get_yaw()
            
            # Draw direction indicator
            dx = agent.radius * 1.5 * np.cos(yaw)
            dy = agent.radius * 1.5 * np.sin(yaw)
            ax.arrow(
                agent.pose.x, agent.pose.y,
                dx, dy,
                head_width=0.1, head_length=0.1,
                fc='darkblue', ec='darkblue',
                zorder=11
            )
            
            # Draw camera field of view (cone) if camera is active
            if hasattr(agent, 'camera_active') and agent.camera_active:
                fov_angle = getattr(agent, 'camera_fov', np.pi / 4)
                fov_length = getattr(agent, 'camera_range', 2.0)
                
                # Draw FOV cone
                for angle_offset in [-fov_angle/2, fov_angle/2]:
                    angle = yaw + angle_offset
                    end_x = agent.pose.x + fov_length * np.cos(angle)
                    end_y = agent.pose.y + fov_length * np.sin(angle)
                    ax.plot(
                        [agent.pose.x, end_x],
                        [agent.pose.y, end_y],
                        'r--', linewidth=1.5, alpha=0.6, zorder=9
                    )
                
                # Draw arc to show FOV area
                theta_start = np.degrees(yaw - fov_angle/2)
                theta_end = np.degrees(yaw + fov_angle/2)
                arc = mpatches.Arc(
                    (agent.pose.x, agent.pose.y),
                    2 * fov_length * 0.3, 2 * fov_length * 0.3,
                    angle=0, theta1=theta_start, theta2=theta_end,
                    color='red', linewidth=1, alpha=0.4, linestyle='--'
                )
                ax.add_patch(arc)
    
    # Add width annotations to show narrow/broad sections
    # This helps visualize the varying width
    width_annotations = [
        (1.0, 3.0, "Broad\nEntrance", "2.9m"),
        (5.0, 2.75, "Narrow\nPassage", "0.4m"),
        (11.0, 3.0, "Widening\nExit", "1.2m"),
    ]
    for x, y, label, width in width_annotations:
        ax.annotate(f'{label}\n({width})', xy=(x, y), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, ha='left', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                   zorder=10)
    
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Custom Narrow World - 4 Agents with Camera Feeds\n(Broad Entrance → Narrow Passage → Wide Exit)', 
                fontsize=14, fontweight='bold')
    ax.grid(False)  # We have our custom grid
    
    plt.tight_layout()
    return fig, ax


def run_simulation(save_image=False, image_filename="narrow_world.png"):
    """Main function to create and run the simulation.
    
    Args:
        save_image: If True, save the visualization instead of showing it
        image_filename: Filename to save the image if save_image is True
    """
    
    print("Creating world with obstacles...")
    try:
        world = create_obstacle_world()  # Use this for filled polygon obstacles
        # world = create_narrow_world()  # Or use this for narrow corridor
    except Exception as e:
        print(f"Error creating world: {e}")
        # Try to disable GUI if it's causing issues
        import os
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        world = create_obstacle_world()  # Retry after disabling GUI
    
    print("Adding 4 agents with camera feeds at entrance...")
    agents = add_agents_with_cameras(world, num_agents=4)
    
    print("\nSetting up ROS 1 topics for robots...")
    setup_result = setup_ros_topics(world, agents)
    if ROS_AVAILABLE:
        node_name, publishers, velocity_commands = setup_result
    else:
        node_name, publishers, velocity_commands = setup_result
    
    print("Setting up visualization...")
    # Use pyrobosim's built-in GUI system
    gui_app = None
    
    if not save_image:
        try:
            # Create QApplication first (required for Qt widgets)
            from PySide6.QtWidgets import QApplication
            import sys as sys_module
            
            # Get or create QApplication instance
            app_qt = QApplication.instance()
            if app_qt is None:
                app_qt = QApplication(sys_module.argv)
            
            # Now import and create pyrobosim's GUI
            from pyrobosim.gui.main import PyRoboSimMainWindow
            from matplotlib.patches import Polygon as MPLPolygon
            print("Using pyrobosim's built-in GUI...")
            gui_app = PyRoboSimMainWindow(world)
            gui_app.show()
            print("PyRoboSim GUI started successfully")
            
            # Add simple polygon obstacles to the GUI canvas
            # These are just filled shapes drawn directly on the canvas
            try:
                if hasattr(gui_app, 'canvas') and hasattr(gui_app.canvas, 'axes') and hasattr(world, 'obstacle_polygons'):
                    obstacles = world.obstacle_polygons
                    print(f"  Adding {len(obstacles)} obstacle polygons to GUI...")
                    
                    # Store obstacle data in canvas
                    gui_app.canvas.obstacle_data = obstacles
                    
                    # Create a function to draw obstacles
                    def draw_obstacles():
                        """Draw filled polygon and circle obstacles on the canvas"""
                        from matplotlib.patches import Circle as MPLCircle
                        
                        if not hasattr(gui_app.canvas, 'obstacle_patches'):
                            gui_app.canvas.obstacle_patches = []
                        
                        # Remove old patches
                        for patch in gui_app.canvas.obstacle_patches:
                            try:
                                patch.remove()
                            except:
                                pass
                        gui_app.canvas.obstacle_patches = []
                        
                        # Draw new patches
                        for obs in obstacles:
                            if obs.get('type') == 'polygon':
                                # Draw polygon obstacle
                                obstacle_patch = MPLPolygon(
                                    obs['coords'],
                                    facecolor=obs['color'],
                                    edgecolor='black',
                                    linewidth=1.5,
                                    alpha=0.95,
                                    zorder=100  # Very high z-order to appear on top
                                )
                            elif obs.get('type') == 'circle':
                                # Draw circle obstacle
                                obstacle_patch = MPLCircle(
                                    obs['center'],
                                    obs['radius'],
                                    facecolor=obs['color'],
                                    edgecolor='black',
                                    linewidth=1.5,
                                    alpha=0.95,
                                    zorder=100  # Very high z-order to appear on top
                                )
                            else:
                                continue
                            
                            gui_app.canvas.axes.add_patch(obstacle_patch)
                            gui_app.canvas.obstacle_patches.append(obstacle_patch)
                    
                    # Draw obstacles initially
                    draw_obstacles()
                    
                    # Hook into canvas updates to redraw obstacles
                    if hasattr(gui_app.canvas, 'draw_signal'):
                        gui_app.canvas.draw_signal.connect(draw_obstacles)
                    
                    # Also hook into show_rooms
                    if hasattr(gui_app.canvas, 'show_rooms'):
                        original_show_rooms = gui_app.canvas.show_rooms
                        def show_rooms_with_obstacles():
                            original_show_rooms()
                            draw_obstacles()
                        gui_app.canvas.show_rooms = show_rooms_with_obstacles
                    
                    # Force canvas redraw
                    if hasattr(gui_app.canvas, 'draw'):
                        gui_app.canvas.draw()
                    
                    print(f"  ✓ Added {len(obstacles)} filled polygon obstacles to GUI")
                elif not hasattr(world, 'obstacle_polygons'):
                    print("  Note: No obstacle_polygons found in world (use create_obstacle_world)")
            except Exception as e:
                print(f"  Warning: Could not add obstacle visualizations: {e}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f"Could not start pyrobosim GUI: {e}")
            import traceback
            traceback.print_exc()
            gui_app = None
            app_qt = None
    
    print("\nWorld created successfully!")
    print(f"  - Rooms: {len(world.rooms)}")
    if len(world.rooms) > 1:
        print(f"    Main room: {world.rooms[0].name}")
        print(f"    Obstacle rooms: {len(world.rooms) - 1}")
        for i, room in enumerate(world.rooms[1:6]):  # Show first 5 obstacles
            print(f"      {i+1}. {room.name} - has viz_patch: {hasattr(room, 'viz_patch') and room.viz_patch is not None}")
        if len(world.rooms) > 6:
            print(f"      ... and {len(world.rooms) - 6} more")
    obstacles_count = len(world.obstacles) if hasattr(world, 'obstacles') else 0
    print(f"  - Obstacles: {obstacles_count}")
    print(f"  - Locations: {len(world.locations)}")
    print(f"  - Agents: {len(agents)}")
    print("\nEach agent has:")
    print("  - Camera sensor activated")
    print("  - Position at entrance (left side)")
    print("  - Facing into the narrow path")
    
    if ROS_AVAILABLE and node_name is not None:
        print("\nROS 1 Topics:")
        print("  - Publishing odometry, pose, and camera feeds")
        print("  - Use 'rostopic list' to see all topics")
        print("  - Use 'rostopic echo /agent_X/odom' to see robot data")
        print("  - Use 'rostopic echo /agent_X/camera/image_raw' to see camera feed")
        print("\n  To control robots, publish to cmd_vel topics:")
        print("  - rostopic pub /agent_1/cmd_vel geometry_msgs/Twist \"{linear: {x: 0.5}, angular: {z: 0.0}}\"")
    
    # Save or show the visualization
    if save_image:
        # For saving, use matplotlib to create an image if available
        if MATPLOTLIB_AVAILABLE:
            fig, ax = visualize_world_with_grid(world, agents)
            plt.savefig(image_filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"\nVisualization saved to {image_filename}")
        else:
            print("Matplotlib not available for saving images")
    else:
        # Use pyrobosim's GUI for interactive display
        if gui_app is not None:
            print("\nStarting real-time simulation with pyrobosim GUI...")
            print("  - Send velocity commands via ROS topics to move robots")
            print("  - PyRoboSim GUI will update in real-time")
            print("  - Close the GUI window to stop")
            
            if ROS_AVAILABLE and node_name is not None:
                # Set up ROS spinner in background thread to make topics discoverable
                ros_spinner_thread = None
                try:
                    def ros_spinner_worker():
                        """Run ROS spinner in background thread for discovery"""
                        try:
                            print("ROS 1 spinner started - topics should now be discoverable")
                            # Use non-blocking spinner that processes callbacks periodically
                            while not rospy.is_shutdown():
                                rospy.rostime.wallsleep(0.1)  # Sleep 100ms
                                # Process callbacks (this makes topics stay registered)
                                try:
                                    # This processes pending callbacks
                                    rospy.rostime.wallsleep(0.001)
                                except:
                                    pass
                        except rospy.ROSInterruptException:
                            pass
                        except Exception as e:
                            print(f"ROS spinner error: {e}")
                    
                    # Start ROS spinner in background thread
                    ros_spinner_thread = threading.Thread(target=ros_spinner_worker, daemon=True)
                    ros_spinner_thread.start()
                    print("ROS 1 spinner thread started")
                    
                    # Re-publish messages now that spinner is running to ensure topics register
                    print("Re-publishing initial messages to register topics...")
                    try:
                        # Verify rospy is ready before publishing
                        try:
                            test_time = rospy.Time.now()
                            print(f"  ✓ rospy.Time ready: {test_time.secs}.{test_time.nsecs}")
                        except Exception as time_err:
                            print(f"  ⚠️  rospy.Time not ready: {time_err}")
                            print("  This might mean roscore isn't running or node isn't fully initialized")
                            # Try to use zero time instead
                            try:
                                test_time = rospy.Time(0, 0)
                                print("  Using zero time as fallback")
                            except:
                                pass
                        
                        # Re-publish all messages
                        publish_robot_data(node_name, publishers, world, agents)
                        print("  ✓ Re-published messages for all robots")
                    except Exception as e:
                        print(f"  ⚠️  Could not re-publish messages: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Give ROS 1 time to discover and advertise topics
                    # Topics need time to register with the master after publishing
                    import time
                    time.sleep(2.0)  # Increased delay to ensure topics are registered
                    
                    # Verify topics are visible
                    try:
                        import xmlrpc.client
                        import os
                        # Get master URI from environment or use default
                        master_uri = os.environ.get('ROS_MASTER_URI', 'http://localhost:11311')
                        master_proxy = xmlrpc.client.ServerProxy(master_uri)
                        
                        # Try to get published topics
                        code, msg, topics = master_proxy.getPublishedTopics('/', '')
                        if code == 1:
                            # Get all topics
                            all_topic_names = [t[0] for t in topics]
                            
                            # Filter for robot topics
                            robot_topics = [t for t in all_topic_names if any(f'/{agent.name}/' in t for agent in agents)]
                            
                            print(f"✓ Topics check: Found {len(all_topic_names)} total topics, {len(robot_topics)} robot topics")
                            
                            if robot_topics:
                                print("  ✓ Robot topics are visible:")
                                for topic_name in robot_topics[:5]:
                                    print(f"    - {topic_name}")
                                if len(robot_topics) > 5:
                                    print(f"    ... and {len(robot_topics) - 5} more")
                            else:
                                print("  ⚠️  No robot topics found yet.")
                                print("      This is normal - topics may take a moment to register.")
                                print("      Try running 'rostopic list' in another terminal to verify.")
                                print(f"      Expected topics: /agent_1/odom, /agent_1/pose, /agent_1/camera/image_raw, etc.")
                            
                            print("\n  Run 'rostopic list | grep agent' to see all agent topics")
                        else:
                            print(f"⚠️  Could not query master for topics: {msg}")
                    except Exception as e:
                        print(f"⚠️  Could not verify topic visibility: {e}")
                        print("  This is not critical - topics may still be working.")
                        print("  Try running 'rostopic list' in another terminal to verify.")
                    
                except Exception as e:
                    print(f"Warning: Could not start ROS spinner thread: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Set up update timer for real-time updates
                try:
                    from PySide6.QtCore import QTimer
                    
                    # Create update function for real-time simulation
                    def update_simulation():
                        # ROS callbacks are handled by the spinner thread
                        # Just update robot poses and publish data
                        
                        # Update robot poses based on velocity commands
                        if velocity_commands:
                            # Check if any commands are non-zero
                            has_movement = any(
                                abs(cmd.get('linear_x', 0)) > 0.001 or abs(cmd.get('angular_z', 0)) > 0.001
                                for cmd in velocity_commands.values()
                            )
                            if has_movement:
                                update_robot_poses(world, agents, velocity_commands, dt=0.05)  # Smaller dt for smoother updates
                        
                        # Publish updated data
                        try:
                            publish_robot_data(node_name, publishers, world, agents)
                        except Exception as pub_error:
                            # Publishing might fail, but continue anyway
                            pass
                        
                        # Process ROS callbacks (non-blocking)
                        try:
                            rospy.rostime.wallsleep(0.001)
                        except:
                            pass
                        
                        # Update pyrobosim GUI - need to call update_robots_plot and redraw
                        if hasattr(gui_app, 'canvas'):
                            # Use pyrobosim's canvas update methods
                            if hasattr(gui_app.canvas, 'update_robots_plot'):
                                gui_app.canvas.update_robots_plot()
                            if hasattr(gui_app.canvas, 'draw_signal'):
                                gui_app.canvas.draw_signal.emit()
                            elif hasattr(gui_app.canvas, 'draw'):
                                gui_app.canvas.draw()
                        elif hasattr(gui_app, 'update_gui'):
                            gui_app.update_gui()
                        elif hasattr(world, 'gui') and world.gui is not None:
                            # Update through world's GUI
                            if hasattr(world.gui, 'canvas'):
                                if hasattr(world.gui.canvas, 'update_robots_plot'):
                                    world.gui.canvas.update_robots_plot()
                                if hasattr(world.gui.canvas, 'draw_signal'):
                                    world.gui.canvas.draw_signal.emit()
                    
                    # Set up timer for updates (20 FPS)
                    timer = QTimer()
                    timer.timeout.connect(update_simulation)
                    timer.start(50)  # Update every 50ms
                    
                    # Run Qt event loop (app_qt already created above)
                    app_qt.exec()
                    
                except Exception as e:
                    print(f"Error setting up real-time updates: {e}")
                    import traceback
                    traceback.print_exc()
                    # Just run the GUI without updates
                    app_qt.exec()
            else:
                # Non-ROS mode - just run GUI (app_qt already created above)
                app_qt.exec()
        else:
            # Fallback to matplotlib if GUI failed
            if MATPLOTLIB_AVAILABLE:
                print("Falling back to matplotlib visualization...")
                fig, ax = visualize_world_with_grid(world, agents)
                plt.show(block=True)
            else:
                print("No visualization available. Use --save to save an image.")
    
    # You can also save the world if needed
    # if hasattr(world, 'to_yaml'):
    #     world.to_yaml("custom_narrow_world.yaml")
    
    return world, agents


def check_ros_master():
    """Check if ROS master is accessible (local setup)."""
    try:
        if ROS_AVAILABLE:
            try:
                master = rospy.get_master()
                master.getSystemState()
                print("✓ Successfully connected to ROS master")
                return True
            except Exception as e:
                print(f"✗ Cannot connect to ROS master: {e}")
                print("  Make sure roscore is running (run 'roscore' in another terminal)")
                return False
        return False
    except Exception as e:
        print(f"Error checking ROS master: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    # Check if we should save instead of show
    save_image = '--save' in sys.argv or '-s' in sys.argv
    
    world, agents = run_simulation(save_image=save_image)
    
    if not save_image:
        print("\nSimulation running. Close the plot window to exit.")
        print("Tip: Use --save or -s flag to save the image instead of displaying it.")
        print("\nNote: If GUI doesn't work, the script will automatically save to narrow_world.png")
    # You can add interactive controls or simulation loop here

