#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script generates a boustrophedon (lawnmower pattern) path within specified 
target polygons defined in a KML/KMZ file, while avoiding designated obstacle 
polygons from the same file. It handles coordinate transformations (WGS84 to UTM), 
path generation at a specified angle and separation, obstacle avoidance by 
splitting the path, and stitching the remaining segments together by navigating 
along obstacle boundaries. The final path, along with the target and obstacle 
polygons, is saved to an output KML/KMZ file.

Authors: Peter Lehnér, Gemini-2.5.Pro-03-25
"""
import sys
import argparse
import numpy as np
from shapely.geometry import Polygon, LineString, MultiPolygon, Point, MultiLineString, MultiPoint, GeometryCollection
from shapely.ops import unary_union, transform
from shapely.affinity import rotate
from lxml import etree
import pyproj
import random
import simplekml
import math
import os # Needed for path manipulation
import zipfile # Needed for KMZ
import io # Needed for reading from zip

# Tolerance for floating point comparisons
TOLERANCE = 1e-9

def read_kml_polygons(kml_file):
    """
    Read ALL polygons from a KML or KMZ file.
    Returns a list of tuples, where each tuple is (name, polygon_geometry).
    """
    doc = None
    # Check file extension
    if kml_file.lower().endswith('.kmz'):
        print(f"Reading KMZ file: {kml_file}")
        try:
            with zipfile.ZipFile(kml_file, 'r') as archive:
                # Find the main KML file within the KMZ
                kml_filename = None
                # Prioritize doc.kml if it exists
                if 'doc.kml' in archive.namelist():
                     kml_filename = 'doc.kml'
                else:
                    # Look for the first *.kml file
                    for name in archive.namelist():
                        if name.lower().endswith('.kml'):
                            kml_filename = name
                            break
                
                if kml_filename:
                    print(f"  Found KML content in: {kml_filename}")
                    doc = archive.read(kml_filename)
                else:
                    raise ValueError("No .kml file found within the KMZ archive.")
        except zipfile.BadZipFile:
             raise ValueError(f"Error: Input file '{kml_file}' is not a valid KMZ file.")
        except Exception as e:
             raise ValueError(f"Error reading KMZ file '{kml_file}': {e}")

    elif kml_file.lower().endswith('.kml'):
        print(f"Reading KML file: {kml_file}")
        try:
            with open(kml_file, 'rb') as f:
                doc = f.read()
        except Exception as e:
             raise ValueError(f"Error reading KML file '{kml_file}': {e}")
    else:
         raise ValueError(f"Error: Input file '{kml_file}' must be a .kml or .kmz file.")

    if doc is None:
         raise ValueError(f"Error: Could not read KML content from '{kml_file}'.")

    # Parse the KML content (from either KML or extracted from KMZ)
    try:
        root = etree.fromstring(doc)
    except etree.XMLSyntaxError as e:
        raise ValueError(f"Error parsing KML content from '{kml_file}': {e}")
        
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    
    placemark_elements = root.findall('.//kml:Placemark', namespaces=ns)
    
    all_polygons_data = [] # Store tuples of (name, polygon_geometry)
    unnamed_count = 0

    for placemark in placemark_elements:
        name_elem = placemark.find('./kml:name', namespaces=ns)
        poly_elem = placemark.find('./kml:Polygon', namespaces=ns)
        
        if poly_elem is not None:
            coords_elem = poly_elem.find('.//kml:coordinates', namespaces=ns)
            if coords_elem is not None:
                coords_text = coords_elem.text.strip()
                coords = []
                for coord in coords_text.split():
                    parts = coord.split(',')
                    if len(parts) >= 2:
                        lon, lat = float(parts[0]), float(parts[1])
                        coords.append((lon, lat))
                
                if len(coords) >= 3:
                    polygon = Polygon(coords)
                    if polygon.is_valid:
                        if name_elem is not None and name_elem.text:
                            name = name_elem.text
                        else:
                            unnamed_count += 1
                            name = f"UnnamedPolygon_{unnamed_count}"
                            print(f"Warning: Found unnamed polygon, assigning name: {name}")
                        all_polygons_data.append((name, polygon))

    if not all_polygons_data:
        print(f"Warning: No valid polygons found in the KML content of '{kml_file}'.")
        # Allow returning empty list instead of raising error?
        # raise ValueError("No valid polygons found in KML file.") 
    
    print(f"Read {len(all_polygons_data)} total polygons from KML content.")
    return all_polygons_data

def generate_boustrophedon_path(main_polygon_deg, separation_meters=0.3, angle_deg=90.0):
    """
    Generate a boustrophedon path at a specified angle and separation.
    Projects to UTM, rotates, calculates path, rotates back.
    Returns the path in PROJECTED coordinates (UTM), plus the centroid and transformers.
    """
    # --- Coordinate System Setup & Projection --- 
    crs_deg = "EPSG:4326"
    
    # Reverted: CRS determination should be handled by the caller if multiple polygons are involved
    # The function itself determines CRS based *only* on the input geometry
    if isinstance(main_polygon_deg, MultiPolygon):
        centroid_deg = main_polygon_deg.centroid
    elif isinstance(main_polygon_deg, Polygon):
        centroid_deg = main_polygon_deg.centroid
    else:
        raise TypeError(f"Expected Polygon or MultiPolygon, got {type(main_polygon_deg)}")

    utm_zone = int((centroid_deg.x + 180) / 6) + 1
    hemisphere_code = 6 if centroid_deg.y >= 0 else 7
    crs_proj = f"EPSG:32{hemisphere_code}{utm_zone:02d}" 
    # print(f"Debug (generate_path): Using projected CRS {crs_proj} based on input centroid")

    transformer_to_proj = pyproj.Transformer.from_crs(crs_deg, crs_proj, always_xy=True)
    transformer_to_deg = pyproj.Transformer.from_crs(crs_proj, crs_deg, always_xy=True)

    main_geometry_proj = transform(transformer_to_proj.transform, main_polygon_deg)
    # Use bounds of potentially MultiPolygon
    bounds_proj_for_width = main_geometry_proj.bounds 
    width_proj = bounds_proj_for_width[2] - bounds_proj_for_width[0] 
    centroid_proj = main_geometry_proj.centroid

    # --- Rotation Setup --- 
    math_angle_deg = (90.0 - angle_deg) % 360.0 

    # --- Rotate Polygon for Horizontal Scan --- 
    geometry_rotated = rotate(main_geometry_proj, -math_angle_deg, origin=centroid_proj)
    
    # --- Generate Path on Rotated Polygon --- 
    bounds_rot = geometry_rotated.bounds
    min_x_rot, min_y_rot, max_x_rot, max_y_rot = bounds_rot

    height_rot = max_y_rot - min_y_rot
    separation_meters = abs(separation_meters) if separation_meters > 1e-6 else 0.1 

    clipped_lines_rot = []
    y_coords_rot = np.linspace(min_y_rot + separation_meters / 2, max_y_rot - separation_meters / 2, num=max(1, int(height_rot / separation_meters)))
    if not y_coords_rot.size: # Handle case where height < separation
        y_coords_rot = np.array([(min_y_rot + max_y_rot) / 2])
        

    for current_y_rot in y_coords_rot:
        scan_line_rot = LineString([(min_x_rot - width_proj*1.1, current_y_rot), 
                                    (max_x_rot + width_proj*1.1, current_y_rot)])

        clipped_geom_rot = geometry_rotated.buffer(0).intersection(scan_line_rot) 
        if clipped_geom_rot.is_empty:
            continue

        segments_rot = []
        if isinstance(clipped_geom_rot, LineString):
            if clipped_geom_rot.length > TOLERANCE:
                segments_rot = [clipped_geom_rot]
        elif isinstance(clipped_geom_rot, MultiLineString):
            segments_rot = [s for s in clipped_geom_rot.geoms if s.length > TOLERANCE]
        
        if segments_rot:
             segments_rot.sort(key=lambda s: s.coords[0][0]) 
             clipped_lines_rot.append(segments_rot)
             
    if not clipped_lines_rot:
         return LineString(), centroid_proj, transformer_to_proj, transformer_to_deg

    # --- Assemble Path in Rotated Coordinates --- 
    final_path_coords_rot = []
    direction = True 
    last_processed_endpoint_coord_rot = None

    for segments_on_line_rot in clipped_lines_rot:
        if not segments_on_line_rot:
            continue
        
        connection_coord_rot = segments_on_line_rot[0].coords[0] if direction else segments_on_line_rot[-1].coords[-1]
        current_endpoint_coord_rot = segments_on_line_rot[-1].coords[-1] if direction else segments_on_line_rot[0].coords[0]

        if last_processed_endpoint_coord_rot:
            if Point(last_processed_endpoint_coord_rot).distance(Point(connection_coord_rot)) > TOLERANCE:
                final_path_coords_rot.append(last_processed_endpoint_coord_rot)
                final_path_coords_rot.append(connection_coord_rot)
            elif not final_path_coords_rot or Point(final_path_coords_rot[-1]).distance(Point(connection_coord_rot)) > TOLERANCE:
                final_path_coords_rot.append(connection_coord_rot)

        elif not final_path_coords_rot: 
             final_path_coords_rot.append(connection_coord_rot)

        coords_this_line_rot = []
        if direction: 
            for k, segment in enumerate(segments_on_line_rot):
                coords = list(segment.coords)
                if not coords_this_line_rot or Point(coords_this_line_rot[-1]).distance(Point(coords[0])) > TOLERANCE:
                     coords_this_line_rot.append(coords[0])
                coords_this_line_rot.extend(coords[1:])
        else: 
            for k, segment in enumerate(reversed(segments_on_line_rot)):
                coords = list(segment.coords)[::-1] 
                if not coords_this_line_rot or Point(coords_this_line_rot[-1]).distance(Point(coords[0])) > TOLERANCE:
                     coords_this_line_rot.append(coords[0])
                coords_this_line_rot.extend(coords[1:])
                
        if final_path_coords_rot and Point(final_path_coords_rot[-1]).distance(Point(coords_this_line_rot[0])) < TOLERANCE:
             final_path_coords_rot.extend(coords_this_line_rot[1:])
        else:
             final_path_coords_rot.extend(coords_this_line_rot)
             
        last_processed_endpoint_coord_rot = current_endpoint_coord_rot
        direction = not direction

    if last_processed_endpoint_coord_rot and (not final_path_coords_rot or Point(final_path_coords_rot[-1]).distance(Point(last_processed_endpoint_coord_rot)) > TOLERANCE):
         final_path_coords_rot.append(last_processed_endpoint_coord_rot)

    if len(final_path_coords_rot) < 2:
        return LineString(), centroid_proj, transformer_to_proj, transformer_to_deg
    
    # --- Clean path in rotated coordinates first ---
    cleaned_coords_rot = [final_path_coords_rot[0]]
    for i in range(1, len(final_path_coords_rot)):
        if Point(final_path_coords_rot[i]).distance(Point(cleaned_coords_rot[-1])) > TOLERANCE:
            cleaned_coords_rot.append(final_path_coords_rot[i])
            
    if len(cleaned_coords_rot) < 2:
         return LineString(), centroid_proj, transformer_to_proj, transformer_to_deg
         
    final_path_rot = LineString(cleaned_coords_rot)
    final_path_rot = final_path_rot.simplify(TOLERANCE * 10, preserve_topology=True)
    if final_path_rot.is_empty or len(final_path_rot.coords) < 2:
         return LineString(), centroid_proj, transformer_to_proj, transformer_to_deg
         
    # --- Rotate Path Back to Original Orientation (Projected) --- 
    final_path_proj = rotate(final_path_rot, math_angle_deg, origin=centroid_proj)
    
    # --- Final Cleanup in Projected Coordinates --- 
    if final_path_proj.is_empty or len(final_path_proj.coords) < 2:
         return LineString(), centroid_proj, transformer_to_proj, transformer_to_deg
         
    cleaned_coords_proj = [final_path_proj.coords[0]]
    for i in range(1, len(final_path_proj.coords)):
        if Point(final_path_proj.coords[i]).distance(Point(cleaned_coords_proj[-1])) > TOLERANCE:
             cleaned_coords_proj.append(final_path_proj.coords[i])
             
    if len(cleaned_coords_proj) < 2:
         return LineString(), centroid_proj, transformer_to_proj, transformer_to_deg

    final_path_clean_proj = LineString(cleaned_coords_proj)

    return final_path_clean_proj, centroid_proj, transformer_to_proj, transformer_to_deg

def split_path_by_obstacles(base_path_proj, obstacles_proj, separation_meters=0.3):
    """
    Splits a base path LineString (PROJECTED) by subtracting obstacles (PROJECTED).
    Returns a list of LineString tracks (PROJECTED) that are outside obstacles.
    """
    if not base_path_proj.is_valid or base_path_proj.is_empty:
        print("Warning: Base path is invalid or empty for splitting.")
        return []
    if not obstacles_proj:
        return [base_path_proj]

    valid_obstacles_proj = [obs for obs in obstacles_proj if obs.is_valid and not obs.is_empty]
    if not valid_obstacles_proj:
        # print("No valid projected obstacles to split by.") # Removed
        return [base_path_proj]

    # --- Difference Operation (Projected Coords) ---
    # Use a small buffer based purely on TOLERANCE for splitting
    SPLIT_BUFFER = TOLERANCE * 50 # Drastically reduced buffer
    try:
         obstacles_union_proj = unary_union(valid_obstacles_proj)
         buffered_obstacles_union_proj = obstacles_union_proj.buffer(SPLIT_BUFFER, join_style=2) 
    except Exception as e:
         print(f"Error buffering/unioning obstacles: {e}. Proceeding without splitting.")
         return [base_path_proj]

    try:
        path_after_difference = base_path_proj.difference(buffered_obstacles_union_proj)
    except Exception as e:
        print(f"Error performing difference operation: {e}. Proceeding without splitting.")
        return [base_path_proj]

    # --- Extract Valid LineString Tracks --- 
    tracks_proj = []
    if path_after_difference.is_empty:
        pass # print("Warning: Path is empty after difference operation.") # Removed
    elif isinstance(path_after_difference, LineString):
        if path_after_difference.length > TOLERANCE:
            tracks_proj = [path_after_difference]
    elif isinstance(path_after_difference, MultiLineString):
        tracks_proj = sorted(
            [line for line in path_after_difference.geoms if line.length > TOLERANCE],
            key=lambda line: base_path_proj.project(Point(line.coords[0])) if line.coords else float('inf')
        )
    elif isinstance(path_after_difference, GeometryCollection):
        # print("Info: Difference resulted in GeometryCollection. Extracting and sorting LineStrings.") # Removed
        lines = []
        for geom in path_after_difference.geoms:
            if isinstance(geom, LineString):
                if geom.length > TOLERANCE:
                     lines.append(geom)
            elif isinstance(geom, MultiLineString):
                 lines.extend([line for line in geom.geoms if line.length > TOLERANCE])
        tracks_proj = sorted(
             lines,
             key=lambda line: base_path_proj.project(Point(line.coords[0])) if line.coords else float('inf')
        )
    # else:
        # print(f"Warning: Unexpected geometry type after difference: {type(path_after_difference)}. No tracks generated.") # Removed

    if not tracks_proj:
        # print("Warning: No valid tracks found after splitting by obstacles.") # Removed
        return []

    # --- Clean Tracks (remove duplicate consecutive points) ---
    cleaned_tracks_proj = []
    for track in tracks_proj:
        if not track.coords: continue
        cleaned_coords = [track.coords[0]]
        for i in range(1, len(track.coords)):
            if Point(track.coords[i]).distance(Point(cleaned_coords[-1])) > TOLERANCE:
                cleaned_coords.append(track.coords[i])
        if len(cleaned_coords) >= 2:
            cleaned_tracks_proj.append(LineString(cleaned_coords))
            
    # print(f"Split into {len(cleaned_tracks_proj)} cleaned tracks (projected).") # Removed
    return cleaned_tracks_proj

def _find_intersection_points(path_proj, obstacles_proj):
    """
    Finds points where a LineString intersects obstacle boundaries.
    Inputs path and obstacles must be in the same PROJECTED CRS.
    Returns a list of coordinate tuples (PROJECTED) for intersection points.
    """
    intersection_points_coords = [] 
    if not path_proj.is_valid or path_proj.is_empty or not obstacles_proj:
        return intersection_points_coords 

    valid_obstacles_proj = [obs for obs in obstacles_proj 
                           if isinstance(obs, (Polygon, MultiPolygon)) and obs.is_valid and not obs.is_empty]
    if not valid_obstacles_proj: 
        # print("No valid projected obstacles provided for intersection check.") # Removed
        return intersection_points_coords
    
    # print(f"Finding intersections between path and {len(valid_obstacles_proj)} projected obstacles...") # Removed

    try:
        obstacle_boundaries = unary_union([obs.boundary for obs in valid_obstacles_proj])
    except Exception as e:
        print(f"Error getting obstacle boundaries: {e}. Cannot find intersections.")
        return []

    if obstacle_boundaries.is_empty:
        # print("Obstacle boundaries are empty. No intersections possible.") # Removed
        return []

    intersection = path_proj.intersection(obstacle_boundaries)

    if intersection.is_empty:
        # print("No intersection points found.") # Removed
        return []
        
    points_to_add = []
    if isinstance(intersection, Point):
        points_to_add.append((intersection.x, intersection.y))
    elif isinstance(intersection, MultiPoint):
        for point in intersection.geoms:
             points_to_add.append((point.x, point.y))
    elif isinstance(intersection, (LineString, MultiLineString, GeometryCollection)):
        # If intersection is linear/complex, extract representative points (e.g., start/end of lines)
        # This handles cases where the path runs *along* a boundary
        # print(f"Warning: Intersection resulted in complex geometry ({type(intersection)}). Extracting points from components.") # Removed
        if isinstance(intersection, GeometryCollection):
            geoms = intersection.geoms
        else: # LineString or MultiLineString
            geoms = getattr(intersection, 'geoms', [intersection]) # Handle single LineString
            
        for geom in geoms:
            if isinstance(geom, Point):
                 points_to_add.append((geom.x, geom.y))
            elif isinstance(geom, LineString) and geom.coords:
                 # Add start and end points of linear intersections
                 points_to_add.append(geom.coords[0])
                 points_to_add.append(geom.coords[-1])
                 # Optionally add interpolated points if needed, but start/end usually suffice
            elif isinstance(geom, MultiPoint):
                 for point in geom.geoms:
                      points_to_add.append((point.x, point.y))

    # --- Filter out points that are very close to original path vertices ---
    # This avoids adding points that were already part of the path definition
    original_path_points = { (round(p[0], 7), round(p[1], 7)) for p in path_proj.coords }
    
    for pt_coord in points_to_add:
         intersection_points_coords.append(pt_coord)
            
   
    # Remove duplicates from the collected points using rounding
    unique_intersection_points = []
    seen_coords = set()
    for pt_coord in intersection_points_coords:
        rounded_coord = (round(pt_coord[0], 7), round(pt_coord[1], 7))
        if rounded_coord not in seen_coords:
            # Further check: ensure point actually lies on *an* obstacle boundary
            # Increase tolerance slightly for this check
            is_on_boundary = False
            for obs_bdy in getattr(obstacle_boundaries, 'geoms', [obstacle_boundaries]):
                 if obs_bdy.distance(Point(pt_coord)) < (TOLERANCE * 10):
                      is_on_boundary = True
                      break
            if is_on_boundary:
                 unique_intersection_points.append(pt_coord)
                 seen_coords.add(rounded_coord)
            # else:
            #      print(f"Debug: Point {pt_coord} from intersection not close enough to final boundary, filtering out.") # Removed
           
    # print(f"Found {len(unique_intersection_points)} unique intersection points (projected).") # Removed
    # RETURN PROJECTED POINTS
    return unique_intersection_points

# =======================================
# RDP Simplification Helpers START
# =======================================

def perpendicular_distance(point, line_start, line_end):
    """Calculates the perpendicular distance from a point to a line segment defined by start and end points."""
    px, py = point
    ax, ay = line_start
    bx, by = line_end

    # Vector from A to B
    vx = bx - ax
    vy = by - ay
    # Vector from A to P
    wx = px - ax
    wy = py - ay

    mag_v_sq = vx**2 + vy**2
    if mag_v_sq < TOLERANCE**2: # Avoid division by zero if start == end
        return math.sqrt(wx**2 + wy**2)

    # Parameter t representing projection of AP onto AB
    dot = wx * vx + wy * vy
    t = dot / mag_v_sq

    if t < 0.0: # Closest point is A
        closest_x, closest_y = ax, ay
    elif t > 1.0: # Closest point is B
        closest_x, closest_y = bx, by
    else: # Closest point is projection
        closest_x = ax + t * vx
        closest_y = ay + t * vy

    # Distance from P to closest point
    dx = px - closest_x
    dy = py - closest_y
    return math.sqrt(dx**2 + dy**2)

def rdp_simplify(point_list, epsilon):
    """Simplifies a list of points using the Ramer-Douglas-Peucker algorithm."""
    if len(point_list) < 3:
        return point_list # Nothing to simplify

    start_point = point_list[0]
    end_point = point_list[-1]
    max_dist = 0.0
    max_index = 0

    # Find the point furthest from the line segment connecting start and end
    for i in range(1, len(point_list) - 1):
        dist = perpendicular_distance(point_list[i], start_point, end_point)
        if dist > max_dist:
            max_dist = dist
            max_index = i

    # If max distance is greater than epsilon, recursively simplify
    if max_dist > epsilon:
        # Recursive call on the first part
        results1 = rdp_simplify(point_list[:max_index+1], epsilon)
        # Recursive call on the second part
        results2 = rdp_simplify(point_list[max_index:], epsilon)

        # Combine the results, excluding the duplicate middle point
        return results1[:-1] + results2
    else:
        # All intermediate points are within tolerance, keep only start and end
        return [start_point, end_point]

# =======================================
# RDP Simplification Helpers END
# =======================================

# =======================================
# Stitching Helper Functions START 
# =======================================

def find_point_index_on_track(point_coord, track, tolerance=1e-6):
    """Finds the index of a point coordinate in a LineString's coords using tolerance."""
    target_point = Point(point_coord)
    min_dist = float('inf')
    best_idx = -1
    if not track.coords:
        return -1
        
    for i, coord in enumerate(track.coords):
        dist = target_point.distance(Point(coord))
        if dist < min_dist:
            min_dist = dist
            best_idx = i
            
    if min_dist < tolerance:
        return best_idx
    else:
        # Try projecting point onto the line to find the closest segment, then check endpoints
        # This might help if the exact point isn't a vertex but lies on a segment
        # This adds complexity, let's stick to vertex check for now.
        # print(f"Debug: Point {point_coord} not found within tolerance {tolerance} on track. Min dist: {min_dist}")
        return -1

def calculate_path_length(coords_list):
    """Calculates the length of a path defined by a list of coordinates (assumes projected CRS)."""
    if not coords_list or len(coords_list) < 2:
        return 0.0
    try:
        return LineString(coords_list).length
    except Exception as e:
        # print(f"Error calculating length for coords list (length {len(coords_list)}): {e}") # Removed
        return 0.0

def find_bridging_obstacle_path(end_point_coord, start_point_coord, obstacle_tracks_proj, tolerance=1e-7):
    """
    Finds the shortest path segment along an obstacle track boundary connecting two points.
    Operates on projected coordinates. Returns the list of coordinates for the shortest path.
    """
    end_pt = Point(end_point_coord)
    start_pt = Point(start_point_coord)

    best_bridge_coords = None
    min_bridge_len = float('inf')
    found_on_any_track = False
    bridging_lookup_tolerance = 1e-5 # Use a larger tolerance for lookups within this function

    for i, obstacle_track in enumerate(obstacle_tracks_proj):
        if not obstacle_track.is_valid or not obstacle_track.coords or len(obstacle_track.coords) < 2:
            continue
        coords = list(obstacle_track.coords)
        
        # Check if the track is closed (first and last points are close)
        # This is important for navigating loops correctly
        is_closed = Point(coords[0]).distance(Point(coords[-1])) < tolerance 
        
        # If the track isn't closed, calculating forward/backward paths is trickier.
        # For simplicity, let's only consider closed obstacle tracks for bridging for now.
        # Open tracks (lines) might represent parts of obstacle boundaries but aren't ideal for routing along.
        if not is_closed and len(coords) > 2 : # Need > 2 points to be potentially closed
             # print(f"Debug: Skipping obstacle track {i} for bridging as it's not closed.")
             continue
             
        n = len(coords)
        # Adjust n if closed: number of segments is n-1
        num_segments = n - 1 if is_closed and n > 1 else n 
        if num_segments == 0: continue # Single point track
        
        idx1 = find_point_index_on_track(end_point_coord, obstacle_track, tolerance=bridging_lookup_tolerance)
        idx2 = find_point_index_on_track(start_point_coord, obstacle_track, tolerance=bridging_lookup_tolerance)

        if idx1 != -1 and idx2 != -1:
            found_on_any_track = True
            current_segment = []
            current_len = 0.0
            
            if idx1 == idx2:
                current_segment = coords[idx1:idx1+1] # Path is just the single point
                current_len = 0.0
            elif not is_closed:
                 # Handle open linestring - just take the direct segment
                 start_index = min(idx1, idx2)
                 end_index = max(idx1, idx2)
                 segment = coords[start_index : end_index+1]
                 if idx1 > idx2: # Reverse if start point comes after end point in list
                      segment = segment[::-1]
                 current_segment = segment
                 current_len = calculate_path_length(current_segment)
            else: # Closed loop logic
                # Path forward
                path_fwd_indices = []
                curr = idx1
                while curr != idx2:
                    path_fwd_indices.append(curr)
                    # Use modulo num_segments (which is n-1 for closed loops)
                    curr = (curr + 1) % num_segments 
                path_fwd_indices.append(idx2)
                segment_fwd = [coords[j] for j in path_fwd_indices]
                len_fwd = calculate_path_length(segment_fwd)

                # Path backward
                path_bwd_indices = []
                curr = idx1
                while curr != idx2:
                    path_bwd_indices.append(curr)
                    # Use modulo num_segments for backward movement too
                    curr = (curr - 1 + num_segments) % num_segments 
                path_bwd_indices.append(idx2)
                segment_bwd = [coords[j] for j in path_bwd_indices]
                len_bwd = calculate_path_length(segment_bwd)

                # Choose shorter path
                if len_fwd <= len_bwd:
                    current_segment = segment_fwd
                    current_len = len_fwd
                else:
                    current_segment = segment_bwd
                    current_len = len_bwd

            # Update best bridge if this one is shorter
            if current_len < min_bridge_len:
                min_bridge_len = current_len
                best_bridge_coords = current_segment
                # print(f"Found new best bridge (len {min_bridge_len:.2f}m) on obstacle track {i}") # Removed

    if not found_on_any_track:
        # print(f"Warning: Could not find *both* points {end_point_coord} and {start_point_coord} on any single obstacle track.") # Removed
        # Maybe try finding closest points on tracks instead? More complex. For now, fail.
        return [] 
    if best_bridge_coords is None:
         # print(f"Warning: Failed to determine shortest bridge path between {end_point_coord} and {start_point_coord} despite finding points.") # Removed
         return []
         
    # print(f"Debug: Found bridge path with {len(best_bridge_coords)} points, length {min_bridge_len:.3f}m")
    return best_bridge_coords

def order_tracks_along_path(tracks_proj, base_path_proj):
    """Sorts tracks based on the projection distance of their start points onto a base path."""
    if not tracks_proj:
        return []
    if not base_path_proj.is_valid or not base_path_proj.coords:
        # print("Warning: Cannot order tracks along an invalid or empty base path. Returning original order.") # Removed
        return tracks_proj 
        
    track_distances = []
    for i, track in enumerate(tracks_proj):
        if not track.is_valid or not track.coords:
            # print(f"Warning: Skipping invalid or empty track {i} during ordering.") # Removed
            distance = float('inf')
        else:
            start_point = Point(track.coords[0])
            try:
                 distance = base_path_proj.project(start_point, normalized=False)
            except Exception as e:
                 print(f"Error projecting start point of track {i}: {e}. Placing track at end.")
                 distance = float('inf')
        track_distances.append({'distance': distance, 'track': track, 'original_index': i})
        
    # Sort by distance along the base path
    track_distances.sort(key=lambda item: item['distance'])
    
    ordered_tracks = []
    skipped_indices = []
    for item in track_distances:
        if item['distance'] == float('inf'):
             skipped_indices.append(item['original_index'])
        else:
             ordered_tracks.append(item['track'])
             
    # if skipped_indices:
    #      print(f"Warning: {len(skipped_indices)} tracks (original indices: {skipped_indices}) could not be ordered and were omitted.") # Removed
    # print(f"Ordered {len(ordered_tracks)} tracks along the base path.") # Removed
    return ordered_tracks

def stitch_path_segments_proj(ordered_tracks_proj, obstacle_tracks_proj):
    """
    Stitches ordered path segments together using the shortest path along obstacle boundaries.
    Operates on projected coordinates.

    Returns:
        A single LineString representing the final stitched path in projected coordinates,
        or None if stitching fails.
    """
    simplify_tolerance = 0.01 # Use 1cm for bridge pre-simplification

    if not ordered_tracks_proj:
        print("Error: No path segments provided to stitch.")
        return None
        
    valid_tracks = [t for t in ordered_tracks_proj if t.is_valid and t.coords]
    if len(valid_tracks) != len(ordered_tracks_proj):
        # print(f"Warning: Filtering out {len(ordered_tracks_proj) - len(valid_tracks)} invalid/empty tracks before stitching.") # Removed
        ordered_tracks_proj = valid_tracks
        if not ordered_tracks_proj:
             print("Error: No valid path segments remaining after filtering.")
             return None
             
    # print(f"Starting stitching process for {len(ordered_tracks_proj)} segments...") # Removed
    final_coords = list(ordered_tracks_proj[0].coords)
    # print(f"  Initial coords from track 0: {len(final_coords)}") # Removed

    for i in range(len(ordered_tracks_proj) - 1):
        # print(f"\n  Processing gap {i}:") # Removed
        # print(f"    Coords before bridge: {len(final_coords)}") # Removed
        track_i = ordered_tracks_proj[i]
        track_i_plus_1 = ordered_tracks_proj[i+1]
        
        if not track_i.coords or not track_i_plus_1.coords:
             print(f"Error: Encountered track with no coords during stitching loop (i={i}). Stopping.")
             # Return what we have stitched so far
             return LineString(final_coords) if len(final_coords) >= 2 else None
             
        end_i = track_i.coords[-1]
        start_i_plus_1 = track_i_plus_1.coords[0]
        # print(f"    Stitching gap between track {i} (ends {end_i}) and track {i+1} (starts {start_i_plus_1})") # Removed
        
        bridge_coords = find_bridging_obstacle_path(end_i, start_i_plus_1, obstacle_tracks_proj)
        # print(f"    Found bridge coords: {len(bridge_coords)}") # Removed

        if len(bridge_coords) >= 3:
            try:
                bridge_line = LineString(bridge_coords)
                simplified_bridge = bridge_line.simplify(simplify_tolerance, preserve_topology=True)
                if simplified_bridge.is_valid and not simplified_bridge.is_empty and len(simplified_bridge.coords) >= 2:
                    new_bridge_coords = list(simplified_bridge.coords)
                    if len(new_bridge_coords) < len(bridge_coords):
                        # print(f"    Pre-simplified bridge from {len(bridge_coords)} to {len(new_bridge_coords)} points.") # Removed
                        bridge_coords = new_bridge_coords
                # else:
                    # print(f"    Warning: Pre-simplification of bridge resulted in invalid/empty geometry.") # Removed
            except Exception as e:
                pass
                # print(f"    Warning: Error during bridge pre-simplification: {e}") # Removed

        if not bridge_coords:
             # If no bridge path found, we cannot reliably continue stitching.
             # Return the path stitched up to this point.
             print(f"Warning: Could not find bridge path between track {i} and {i+1}. Stopping stitching process.")
             # Check if final_coords is valid before returning
             return LineString(final_coords) if len(final_coords) >= 2 else None

        coords_before_bridge_append = len(final_coords)
        if len(bridge_coords) > 0:
             if Point(final_coords[-1]).distance(Point(bridge_coords[0])) < 1e-7: 
                 if len(bridge_coords) > 1: # Add bridge points except the first one
                     final_coords.extend(bridge_coords[1:])
             else: # Gap is too large or points don't match, append the whole bridge
                 # print(f"Warning: Bridge start {bridge_coords[0]} doesn't match current end {final_coords[-1]}. Appending full bridge.") # Removed
                 final_coords.extend(bridge_coords)
        # print(f"    Coords after bridge append: {len(final_coords)} (added {len(final_coords) - coords_before_bridge_append})") # Removed

        coords_before_track_append = len(final_coords)
        next_track_coords = list(track_i_plus_1.coords)
        if len(next_track_coords) > 0:
             # Check distance between last point of current path and first point of next track
             if Point(final_coords[-1]).distance(Point(next_track_coords[0])) < 1e-7:
                 if len(next_track_coords) > 1: # Add next track points except the first one
                     final_coords.extend(next_track_coords[1:])
             else: # Gap is too large or points don't match, append the whole track
                 # print(f"Warning: Next track start {next_track_coords[0]} doesn't match current end {final_coords[-1]}. Appending full track.") # Removed
                 final_coords.extend(next_track_coords)
        # print(f"    Coords after track {i+1} append: {len(final_coords)} (added {len(final_coords) - coords_before_track_append})") # Removed

    # print(f"\nCompleted stitching loop. Coords before final cleanup: {len(final_coords)}") # Removed
    
    if not final_coords:
        print("Error: Final coordinate list is empty after stitching.")
        return None
        
    # --- Final Cleanup and Simplification ---
    # 1. Basic deduplication
    # print(f"Running basic deduplication on {len(final_coords)} points...") # Removed
    deduplicated_coords = [final_coords[0]]
    for j in range(1, len(final_coords)):
        if Point(final_coords[j]).distance(Point(deduplicated_coords[-1])) > TOLERANCE:
            deduplicated_coords.append(final_coords[j])
    # print(f"  Coords after deduplication: {len(deduplicated_coords)}") # Removed
    
    # 2. RDP simplification
    rdp_tolerance = 0.05 # 5cm tolerance for final path simplification
    # print(f"Running RDP simplification with tolerance: {rdp_tolerance}m") # Removed
    if len(deduplicated_coords) >= 3:
        try:
            simplified_coords = rdp_simplify(deduplicated_coords, rdp_tolerance)
            # print(f"  Coords after RDP: {len(simplified_coords)}") # Removed
        except Exception as e:
            print(f"Error during RDP simplification: {e}. Using deduplicated coords.")
            simplified_coords = deduplicated_coords
        else:
            simplified_coords = deduplicated_coords
        
    cleaned_coords = simplified_coords
    # print(f"Cleaned stitched path contains {len(cleaned_coords)} points.") # Removed
    
    if len(cleaned_coords) < 2:
        print("Error: Final stitched path has less than 2 unique points after cleanup.")
        return None
        
    return LineString(cleaned_coords)

# =======================================
# Stitching Helper Functions END 
# =======================================

def create_augmented_obstacle_tracks(obstacles_proj, intersection_points_proj):
    """
    Creates LineString tracks for each obstacle (PROJECTED), including original vertices
    and points (PROJECTED) where the base path intersected the obstacle boundary.
    
    Returns:
        List of LineString objects representing the augmented obstacle tracks (PROJECTED).
    """
    # Reverted to previous version before error-introducing change
    augmented_tracks_proj = []
    intersection_points_geom = [Point(p) for p in intersection_points_proj]
    valid_obstacles_proj = [p for p in obstacles_proj 
                           if isinstance(p, (Polygon, MultiPolygon)) and p.is_valid and not p.is_empty]

    for i, obstacle in enumerate(valid_obstacles_proj):
        obstacle_boundary = obstacle.boundary 
        
        if obstacle_boundary.is_empty or not isinstance(obstacle_boundary, (LineString, MultiLineString)):
            continue

        boundaries_to_process = []
        if isinstance(obstacle_boundary, LineString):
            if obstacle_boundary.coords:
                 boundaries_to_process.append(obstacle_boundary)
        elif isinstance(obstacle_boundary, MultiLineString):
             boundaries_to_process.extend([ls for ls in obstacle_boundary.geoms if ls.coords])
             
        if not boundaries_to_process:
            continue

        for b_idx, boundary_ls in enumerate(boundaries_to_process):
            if not boundary_ls.coords: continue
             
            original_coords = list(boundary_ls.coords)
            relevant_intersections = []
            for pt_geom in intersection_points_geom:
                if boundary_ls.distance(pt_geom) < TOLERANCE:
                     relevant_intersections.append(pt_geom)
                     
            combined_points_geom = {Point(p) for p in original_coords} | set(relevant_intersections)
            
            filtered_points_geom = []
            if combined_points_geom:
                temp_list = list(combined_points_geom)
                if temp_list:
                     filtered_points_geom.append(temp_list[0])
                     for k in range(1, len(temp_list)):
                         is_unique = True
                         for existing_pt in filtered_points_geom:
                              if temp_list[k].distance(existing_pt) < TOLERANCE:
                                   is_unique = False
                                   break
                         if is_unique:
                              filtered_points_geom.append(temp_list[k])

            if len(filtered_points_geom) < 2:
                continue
                
            try:
                point_distances = []
                for pt in filtered_points_geom:
                     distance_along = boundary_ls.project(pt)
                     point_distances.append({'point': pt, 'distance': distance_along})
                     
                point_distances.sort(key=lambda item: item['distance'])
                
                sorted_augmented_coords = [(item['point'].x, item['point'].y) for item in point_distances]

                is_original_closed = Point(original_coords[0]).distance(Point(original_coords[-1])) < TOLERANCE
                is_augmented_closed = Point(sorted_augmented_coords[0]).distance(Point(sorted_augmented_coords[-1])) < TOLERANCE

                if is_original_closed and not is_augmented_closed:
                     sorted_augmented_coords.append(sorted_augmented_coords[0])
                elif not is_original_closed and is_augmented_closed and len(sorted_augmented_coords)>1:
                     sorted_augmented_coords = sorted_augmented_coords[:-1]

                if len(sorted_augmented_coords) >= 2:
                     final_track_coords = [sorted_augmented_coords[0]]
                     for k in range(1, len(sorted_augmented_coords)):
                          if Point(sorted_augmented_coords[k]).distance(Point(final_track_coords[-1])) > TOLERANCE:
                               final_track_coords.append(sorted_augmented_coords[k])
                               
                     if len(final_track_coords) >= 2:
                         augmented_tracks_proj.append(LineString(final_track_coords))

            except Exception as e:
                 print(f"Error creating augmented track component for obstacle {i+1} boundary {b_idx}: {e}")
            
    return augmented_tracks_proj

def save_path_to_kml(final_stitched_path_deg, output_file, output_prefix, reverse_path=False, target_polygons_with_names=None, obstacles_with_names=None):
    """
    Saves the final stitched path, target polygons, and obstacle polygons to a KML or KMZ file.
    Polygons are named based on their original name with suffixes.
    Optionally reverses the path points.
    NO FOLDERS are used in the output KML.
    
    Args:
        final_stitched_path_deg: The single final stitched LineString path (degrees) or None.
        output_file: The name of the KML or KMZ file to save.
        output_prefix: The prefix used for naming the track inside the KML (derived from output filename).
        reverse_path: If True, reverse the order of points in the saved path.
        target_polygons_with_names: List of tuples (name, Polygon/MultiPolygon) for targets (degrees).
        obstacles_with_names: List of tuples (name, Polygon) for obstacles (degrees).
    """
    kml = simplekml.Kml()
    kml.document.name = f"{output_prefix}" 
    
    # --- Define Styles (Keep these) --- 
    target_style = simplekml.Style()
    target_style.polystyle.color = simplekml.Color.changealphaint(77, simplekml.Color.green) # ~30%
    target_style.polystyle.outline = 0
    target_style.linestyle.width = 0

    obstacle_style = simplekml.Style()
    obstacle_style.polystyle.color = simplekml.Color.changealphaint(120, simplekml.Color.red) # ~47%
    obstacle_style.polystyle.outline = 0
    obstacle_style.linestyle.width = 0

    # --- Save Final Stitched Path (directly under document) --- 
    if final_stitched_path_deg and final_stitched_path_deg.is_valid and final_stitched_path_deg.coords:
        track_name = output_prefix
        ls = kml.newlinestring(name=track_name)
        
        coords_list = list(final_stitched_path_deg.coords)
        
        if reverse_path:
            print("Reversing final path points before saving.")
            coords_list.reverse()
            
        ls.coords = [(lon, lat) for lon, lat in coords_list] 
        # --- Update Path Style --- 
        # Match example.kml path style: Yellow, 50% opacity, width 2
        # KML color format is aabbggrr (alpha, blue, green, red)
        # 50% alpha is 80 hex (128 dec)
        # Yellow is 00ffff
        # Combined: 8000ffff
        ls.style.linestyle.color = simplekml.Color.changealphaint(128, simplekml.Color.yellow) # Yellow, 50% alpha
        ls.style.linestyle.width = 2 # Width 2
        # --- End Update Path Style ---
        ls.altitudemode = simplekml.AltitudeMode.clamptoground

        # --- Add Start and Finish Placemarks (directly under document) --- 
        if len(coords_list) >= 1: 
            start_coord = coords_list[0]
            # Ensure point is added directly to kml object
            start_pnt = kml.newpoint(name="start_point", coords=[(start_coord[0], start_coord[1])])
            start_pnt.altitudemode = simplekml.AltitudeMode.clamptoground
            start_pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/S.png'
            
            if len(coords_list) >= 2:
                finish_coord = coords_list[-1]
                if Point(start_coord).distance(Point(finish_coord)) > TOLERANCE:
                    # Ensure point is added directly to kml object
                    finish_pnt = kml.newpoint(name="finish_point", coords=[(finish_coord[0], finish_coord[1])])
                    finish_pnt.altitudemode = simplekml.AltitudeMode.clamptoground
                    finish_pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/F.png' 
                else:
                    start_pnt.name = "Start/Finish" 
            else: 
                 start_pnt.name = "Start/Finish"

    else:
         print("Final stitched path is not available or invalid. Path and points not added to KML.")

    # --- Add Target Polygons (directly under document) --- 
    if target_polygons_with_names: 
        for name, poly_deg in target_polygons_with_names:
             if isinstance(poly_deg, Polygon):
                 polys_to_add = [(name, poly_deg)]
             elif isinstance(poly_deg, MultiPolygon):
                 polys_to_add = [(f"{name}_{j+1}", p) for j, p in enumerate(poly_deg.geoms)]
             else:
                 polys_to_add = []
                 print(f"Warning: Target geometry '{name}' is neither Polygon nor MultiPolygon, skipping.")

             for poly_name, p in polys_to_add:
                 if p.is_valid and p.exterior:
                     # Ensure polygon is added directly to kml object
                     poly_kml = kml.newpolygon(name=f"{poly_name}_target") 
                     poly_kml.outerboundaryis = list(p.exterior.coords)
                     interiors = []
                     for interior in p.interiors:
                          interiors.append(list(interior.coords))
                     if interiors:
                          poly_kml.innerboundaryis = interiors
                     poly_kml.style = target_style 
                     poly_kml.altitudemode = simplekml.AltitudeMode.clamptoground

    # --- Add Obstacle Polygons (directly under document) --- 
    if obstacles_with_names: 
        for name, obs_deg in obstacles_with_names:
            if isinstance(obs_deg, Polygon) and obs_deg.is_valid and obs_deg.exterior:
                 # Ensure polygon is added directly to kml object
                 poly_kml = kml.newpolygon(name=f"{name}_obstacle")
                 poly_kml.outerboundaryis = list(obs_deg.exterior.coords)
                 interiors = []
                 for interior in obs_deg.interiors:
                      interiors.append(list(interior.coords))
                 if interiors:
                      poly_kml.innerboundaryis = interiors
                 poly_kml.style = obstacle_style 
                 poly_kml.altitudemode = simplekml.AltitudeMode.clamptoground
            elif isinstance(obs_deg, MultiPolygon):
                 for j, p in enumerate(obs_deg.geoms):
                      if isinstance(p, Polygon) and p.is_valid and p.exterior:
                           # Ensure polygon is added directly to kml object
                           poly_kml = kml.newpolygon(name=f"{name}_{j+1}_obstacle")
                           poly_kml.outerboundaryis = list(p.exterior.coords)
                           interiors = []
                           for interior in p.interiors:
                                interiors.append(list(interior.coords))
                           if interiors:
                                poly_kml.innerboundaryis = interiors
                           poly_kml.style = obstacle_style
                           poly_kml.altitudemode = simplekml.AltitudeMode.clamptoground
            else:
                 print(f"Warning: Skipping invalid or non-polygon obstacle '{name}'.")

    # Save the KML or KMZ file based on extension
    try:
        if output_file.lower().endswith('.kmz'):
            print(f"Saving as KMZ: {output_file}")
            kml.savekmz(output_file)
        elif output_file.lower().endswith('.kml'):
            print(f"Saving as KML: {output_file}")
            kml.save(output_file)
        else:
             # Default to KML if extension is unknown or missing, maybe add warning?
             print(f"Warning: Unknown output file extension for '{output_file}'. Saving as KML.")
             kml.save(output_file)
    except Exception as e:
        # The exception 'e' might be the NameError if save fails internally
        # Or it could be another error during saving.
        print(f"Error saving KML/KMZ file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate a boustrophedon path for specified KML polygons, avoiding others as obstacles.')
    parser.add_argument('input_kml', help='Path to the input KML file.')
    parser.add_argument('--output', dest='output_kml_file', default=None,
                        help='Path to the output KML file. If not specified, defaults to <input_kml_basename>_path.kml.')
    parser.add_argument('--target', dest='target_polygon_names', action='append', default=[],
                        help='Name of a polygon within the KML to generate paths for. Can be specified multiple times. If none specified, defaults to the largest polygon by area.')
    parser.add_argument('--angle', type=float, default=90.0, 
                        help='Compass angle for the path lines in degrees (0=N, 90=E, 180=S, 270=W). Default is 90.')
    parser.add_argument('--sep', type=float, default=0.3, 
                        help='Separation between path lines in meters. Default is 0.3.')
    parser.add_argument('--reverse', action='store_true', 
                        help='Reverse the order of points in the final output path.')

    # --- Check for missing arguments (only input_kml is mandatory now) ---
    # This check might need refinement if other args become mandatory again
    # Check if only script name is passed or help flag
    if len(sys.argv) == 1 or (len(sys.argv) >= 2 and sys.argv[1] in ('-h', '--help')):
        parser.print_help(sys.stderr)
        sys.exit(1)
    # Check if the first argument exists (likely the input KML)
    # This prevents running parsing if the main input is missing
    elif len(sys.argv) >= 2 and not sys.argv[1].startswith('-') and not os.path.exists(sys.argv[1]):
         # Basic check if the first argument looks like a file that doesn't exist
         print(f"Error: Input KML file not found: {sys.argv[1]}")
         sys.exit(1)

    args = parser.parse_args()

    input_kml = args.input_kml
    target_polygon_names_arg = args.target_polygon_names 
    angle_degrees = args.angle
    separation_in_meters = args.sep
    reverse_final_path = args.reverse 

    # --- Determine Output KML filename --- 
    if args.output_kml_file:
        output_kml_file = args.output_kml_file
    else:
        input_basename = os.path.splitext(os.path.basename(input_kml))[0]
        output_kml_file = f"{input_basename}_path.kml"
        
    # --- Determine Output Prefix for KML track name --- 
    output_prefix = os.path.splitext(os.path.basename(output_kml_file))[0]

    print(f"Input KML: {input_kml}")
    print(f"Output KML: {output_kml_file}")
    print(f"Internal KML Track Name: {output_prefix}") 


    all_polygons_data = read_kml_polygons(input_kml) 
    
    # Keep track of names and geometries together
    target_polygons_with_names = [] 
    obstacles_with_names = [] 
    target_polygon_names_found = [] # Just for the printout
    obstacle_names = [] # Just for the printout

    # --- Target / Obstacle Separation Logic --- 
    if not target_polygon_names_arg: # Default: largest polygon
        print("No target polygons specified via --target. Finding largest polygon by area.")
        if not all_polygons_data:
            print("Error: No polygons found in KML to determine the largest one.")
            sys.exit(1)
            
        largest_poly = None
        largest_area = -1.0
        largest_name = "Unknown"

        # Calculate areas in a suitable projected CRS for better accuracy
        # We need a temporary projection just for area comparison
        # Find a central point for a representative CRS
        valid_polys_for_bounds = [p[1] for p in all_polygons_data if p[1].is_valid]
        if not valid_polys_for_bounds:
             print("Error: No valid polygons available to determine bounds/centroid.")
             sys.exit(1)
             
        combined_bounds = unary_union(valid_polys_for_bounds).bounds
        center_lon = (combined_bounds[0] + combined_bounds[2]) / 2
        center_lat = (combined_bounds[1] + combined_bounds[3]) / 2
        
        temp_utm_zone = int((center_lon + 180) / 6) + 1
        temp_hemisphere = 6 if center_lat >= 0 else 7
        temp_crs_proj = f"EPSG:32{temp_hemisphere}{temp_utm_zone:02d}"
        # print(f"Debug: Using temporary CRS {temp_crs_proj} for area calculation.")
        try:
            temp_transformer = pyproj.Transformer.from_crs("EPSG:4326", temp_crs_proj, always_xy=True)
            can_project_for_area = True
        except pyproj.exceptions.CRSError as e:
            print(f"Warning: Could not create temporary projection {temp_crs_proj} for area calculation ({e}). Area comparison might be less accurate.")
            can_project_for_area = False

        for name, poly_deg in all_polygons_data:
            current_area = 0.0
            if can_project_for_area:
                 try:
                     poly_proj = transform(temp_transformer.transform, poly_deg)
                     current_area = poly_proj.area
                 except Exception as e:
                     print(f"Warning: Could not project/calculate area for polygon '{name}': {e}. Skipping area check for this polygon.")
                     continue # Skip polygon if area calculation fails
            else:
                 # Fallback to degree-based area (less accurate but better than nothing)
                 current_area = poly_deg.area 
                 
            if current_area > largest_area:
                 largest_area = current_area
                 largest_poly = poly_deg
                 largest_name = name


        if largest_poly is None:
            print("Error: Could not determine the largest polygon (possibly due to projection/area errors).")
            sys.exit(1)
            
        area_unit = "sq meters approx" if can_project_for_area else "sq degrees approx"
        print(f"Largest polygon found: '{largest_name}' (Area: {largest_area:.2f} {area_unit})")
        target_polygons_with_names.append((largest_name, largest_poly))
        target_polygon_names_found.append(largest_name)
        
        # All other polygons are obstacles
        for name, poly in all_polygons_data:
             if poly != largest_poly:
                  obstacles_with_names.append((name, poly)) # Store tuple
                  obstacle_names.append(name)

    else: # Specific targets provided
        print(f"Target polygon names specified via --target: {target_polygon_names_arg}")
        target_names_set = set(target_polygon_names_arg)
        # Store found polygons temporarily to allow reordering later
        found_polygons_map = {} 

        for name, poly in all_polygons_data:
            if name in target_names_set:
                if name not in found_polygons_map:
                     found_polygons_map[name] = poly
                # Don't add to obstacles if it's a target
            else:
                obstacles_with_names.append((name, poly)) # Store tuple
                obstacle_names.append(name) # Store obstacle name
                
        # Check if all specified names were found
        found_names_set = set(found_polygons_map.keys())
        missing_names = target_names_set - found_names_set
        if missing_names:
            print(f"Warning: The following specified target polygons were not found in the KML: {list(missing_names)}")
            
        if not found_polygons_map:
             print(f"Error: None of the specified target polygons ({target_polygon_names_arg}) were found in the KML file.")
             sys.exit(1)
        
        # Reorder target polygons based on argument order, storing tuples
        unique_ordered_names_from_args = []
        seen_names = set()
        for name in target_polygon_names_arg:
            if name in found_polygons_map and name not in seen_names:
                 unique_ordered_names_from_args.append(name)
                 seen_names.add(name)

        # Build the final list of target tuples in the specified order
        target_polygons_with_names = [(name, found_polygons_map[name]) for name in unique_ordered_names_from_args]
        target_polygon_names_found = unique_ordered_names_from_args # Update for printout

    print(f"Selected {len(target_polygons_with_names)} target polygons: {target_polygon_names_found}")
    if obstacle_names:
        print(f"Identified {len(obstacles_with_names)} obstacle polygons: {obstacle_names}")
    else:
        print(f"Identified {len(obstacles_with_names)} obstacle polygons.")
    
    # Extract just the geometries for processing, keep the named lists for saving
    target_polygons_deg = [item[1] for item in target_polygons_with_names]
    obstacles_deg = [item[1] for item in obstacles_with_names]
    
    if not target_polygons_deg:
        print("Error: No target polygons selected. Exiting.")
        sys.exit(1)
        
    # --- Determine CRS and DEFINE TRANSFORMERS before use --- 
    crs_deg = "EPSG:4326"
    try:
        combined_targets_for_centroid = unary_union(target_polygons_deg)
        if combined_targets_for_centroid.is_empty:
             print("Warning: Combined target polygons resulted in empty geometry for centroid calculation. Using first target polygon.")
             if target_polygons_deg:
                  centroid_deg = target_polygons_deg[0].centroid
             else: 
                  raise ValueError("No target polygons available for centroid calculation.")
        else:
             centroid_deg = combined_targets_for_centroid.centroid
    except Exception as e:
        print(f"Warning: Could not calculate combined centroid ({e}). Using centroid of the first target polygon.")
        if target_polygons_deg:
             centroid_deg = target_polygons_deg[0].centroid
        else:
              print("Error: Cannot determine centroid because no target polygons are selected.")
              sys.exit(1)
        
    utm_zone = int((centroid_deg.x + 180) / 6) + 1
    hemisphere_code = 6 if centroid_deg.y >= 0 else 7
    crs_proj = f"EPSG:32{hemisphere_code}{utm_zone:02d}" 
    print(f"Using projected CRS: {crs_proj}")

    # --- Create Transformers HERE --- Fix scope issue
    try:
         transformer_to_proj = pyproj.Transformer.from_crs(crs_deg, crs_proj, always_xy=True)
         transformer_to_deg = pyproj.Transformer.from_crs(crs_proj, crs_deg, always_xy=True)
    except pyproj.exceptions.CRSError as e:
         print(f"Error creating coordinate transformers for CRS {crs_proj}: {e}")
         sys.exit(1)
    
    # --- Project Obstacles to UTM --- Now transformers are defined
    obstacles_proj = []
    valid_obstacles_deg = [obs for obs in obstacles_deg if obs.is_valid and not obs.is_empty]
    for i, obs_deg in enumerate(valid_obstacles_deg):
        try:
            obstacles_proj.append(transform(transformer_to_proj.transform, obs_deg))
        except Exception as e:
            print(f"Warning: Could not project obstacle {i+1} to UTM: {e}")
            
    print(f"Projected {len(obstacles_proj)} valid obstacles to UTM.")
    
    # --- Check if target polygons intersect --- 
    targets_intersect = False
    if len(target_polygons_deg) > 1:
        from itertools import combinations
        try:
             target_polygons_proj = [transform(transformer_to_proj.transform, p) for p in target_polygons_deg]
             for poly1_proj, poly2_proj in combinations(target_polygons_proj, 2):
                  if poly1_proj.intersects(poly2_proj):
                       targets_intersect = True
                       print("Target polygons intersect (or touch). Merging them for path generation.")
                       break 
        except Exception as e:
             print(f"Warning: Error projecting/checking intersection of target polygons: {e}. Proceeding assuming they might intersect.")
             targets_intersect = True
             
    # --- Generate Base Path (Projected) --- 
    base_path_proj = None
    # Store centroid returned by generate_boustrophedon_path, might differ if multiple paths run
    path_centroid_proj = None 
    
    if targets_intersect or len(target_polygons_deg) == 1:
        print("Generating base path for merged/single target area...")
        main_geometry_deg = unary_union(target_polygons_deg) 
        if not main_geometry_deg.is_valid or main_geometry_deg.is_empty:
             print("Error: Merged target geometry is invalid or empty. Exiting.")
             sys.exit(1)
        
        # Call generate_boustrophedon_path once 
        # It will use its own internally calculated transformers based on the merged geometry
        base_path_proj, path_centroid_proj, _, path_transformer_to_deg = generate_boustrophedon_path(
            main_geometry_deg, 
            separation_meters=separation_in_meters, 
            angle_deg=angle_degrees
        )
        # Need to capture the specific transformer used for the final transformation back
        # as the global one might be slightly different if centroid shifted
        
    else: # Targets do not intersect, generate paths separately and combine
        print("Generating base paths for separate target areas...")
        all_paths_coords_proj = []
        first_path_transformer_to_deg = None # Store transformer from the first path generated
        
        for i, poly_deg in enumerate(target_polygons_deg): 
            poly_name = target_polygon_names_found[i]
            print(f"  Generating path for target {i+1}/{len(target_polygons_deg)} ('{poly_name}')...")
            
            path_proj, current_centroid_proj, _, current_transformer_to_deg = generate_boustrophedon_path(
                poly_deg, 
                separation_meters=separation_in_meters, 
                angle_deg=angle_degrees
            )
            
            if path_proj and not path_centroid_proj: # Store centroid from first path
                 path_centroid_proj = current_centroid_proj
            if path_proj and not first_path_transformer_to_deg:
                 first_path_transformer_to_deg = current_transformer_to_deg
                 
            if path_proj and not path_proj.is_empty and path_proj.coords:
                 path_coords = list(path_proj.coords)
                 if all_paths_coords_proj and len(path_coords) > 0:
                     all_paths_coords_proj.extend(path_coords)
                 else:
                      all_paths_coords_proj.extend(path_coords)
            else:
                 print(f"Warning: No valid path generated for target '{poly_name}'.")
                 
        if not all_paths_coords_proj:
            print("Error: No path segments generated for any separate target polygon. Exiting.")
            sys.exit(1)
            
        cleaned_combined_coords = []
        if all_paths_coords_proj:
             cleaned_combined_coords.append(all_paths_coords_proj[0])
             for k in range(1, len(all_paths_coords_proj)):
                 if Point(all_paths_coords_proj[k]).distance(Point(cleaned_combined_coords[-1])) > TOLERANCE:
                     cleaned_combined_coords.append(all_paths_coords_proj[k])
                     
        if len(cleaned_combined_coords) < 2:
             print("Error: Combined path has fewer than 2 points after cleaning. Exiting.")
             sys.exit(1)
             
        base_path_proj = LineString(cleaned_combined_coords)
        path_transformer_to_deg = first_path_transformer_to_deg # Use transformer from first segment for final conversion
        print(f"Combined separate paths into a single base path (Points: {len(base_path_proj.coords)}).")

    if base_path_proj is None or not base_path_proj.is_valid or base_path_proj.is_empty:
        print("Error: Base path generation failed. Exiting.")
        sys.exit(1)
        
    # --- The rest of the process uses the generated base_path_proj and projected obstacles --- 
    
    # --- Find Intersection Points (Projected) ---
    intersection_points_proj = _find_intersection_points(base_path_proj, obstacles_proj)
    
    # --- Create Augmented Obstacle Tracks (Projected) ---
    obstacle_tracks_proj = create_augmented_obstacle_tracks(obstacles_proj, intersection_points_proj)
    
    # --- Split the Base Path by Obstacles (Projected) ---
    split_tracks_proj = split_path_by_obstacles(base_path_proj, obstacles_proj, separation_in_meters)
    
    final_stitched_path_proj = None 
    final_path_length_m = 0.0
    final_path_points = 0
    final_path_segments = 0

    if not split_tracks_proj:
        print("Warning: No path tracks generated after splitting by obstacles. Output KML might be empty or incomplete.")
    else:
        # --- Order Tracks Properly (Projected) ---
        ordered_split_tracks_proj = order_tracks_along_path(split_tracks_proj, base_path_proj)
        
        if not ordered_split_tracks_proj:
             print("Error: Failed to order split tracks. Cannot stitch.")
        else:
             # --- Stitch Path Segments (Projected) ---
             final_stitched_path_proj = stitch_path_segments_proj(ordered_split_tracks_proj, obstacle_tracks_proj)

             if not final_stitched_path_proj or final_stitched_path_proj.is_empty:
                  print("Warning: Stitching resulted in an empty or invalid path.")
                  final_stitched_path_proj = None 

    # --- Calculate final metrics IF a stitched path exists --- 
    if final_stitched_path_proj:
        final_path_length_m = final_stitched_path_proj.length
        final_path_points = len(final_stitched_path_proj.coords) if final_stitched_path_proj.coords else 0
        final_path_segments = max(0, final_path_points - 1)

    # --- Transform Results back to Degrees for Output --- 
    final_stitched_path_deg = None
    if final_stitched_path_proj:
        # Use the transformer associated with the final path generation
        # (either from the single call or the first call in the loop)
        if path_transformer_to_deg: 
            try:
                final_stitched_path_deg = transform(path_transformer_to_deg.transform, final_stitched_path_proj)
            except Exception as e:
                print(f"Error transforming final stitched path to degrees: {e}")
                final_stitched_path_deg = None
        else:
            print("Error: Could not determine the correct transformer to convert final path to degrees.")
            
    # --- Save Results to KML --- 
    save_path_to_kml(
        final_stitched_path_deg,
        output_kml_file,
        output_prefix, 
        reverse_final_path,
        target_polygons_with_names, # Pass targets with names
        obstacles_with_names        # Pass obstacles with names
        )

    # --- Final Summary --- 
    print(f"\nProcessing complete.")
    if final_stitched_path_proj: 
        print(f"  Final Path Segments: {final_path_segments}")
        print(f"  Final Path Length:   {final_path_length_m:.2f} meters")
    elif base_path_proj and not split_tracks_proj: 
         print("  Warning: Generated base path was entirely removed by obstacles.")
         print("  No final path generated.")
    elif base_path_proj: 
         if len(split_tracks_proj) == 1 and split_tracks_proj[0].equals(base_path_proj):
              base_len = base_path_proj.length
              base_pts = len(base_path_proj.coords) if base_path_proj.coords else 0
              base_segs = max(0, base_pts - 1)
              print(f"  Path generated (no obstacles intersected or stitching failed/not needed).")
              print(f"  Path Segments: {base_segs}")
              print(f"  Path Length:   {base_len:.2f} meters")
         else: 
              print(f"  Warning: Stitching failed after splitting. Reporting metrics for the base path before splitting.")
              base_len = base_path_proj.length
              base_pts = len(base_path_proj.coords) if base_path_proj.coords else 0
              base_segs = max(0, base_pts - 1)
              print(f"  Base Path Segments: {base_segs}")
              print(f"  Base Path Length:   {base_len:.2f} meters")
              print("  Final output path may be incomplete.")
    else: 
        print("  No final path was generated.")
        
    print(f"  Output KML saved to: {output_kml_file}") 


if __name__ == "__main__":
    main() 