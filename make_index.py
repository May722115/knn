#!/usr/bin/env python3
"""
COMP3323 Assignment 2 - Part 1: Grid Index Construction
  - Data preparation (duplicate elimination)
  - Grid index construction and file I/O
"""

import sys
import time
import argparse
import math

# Bounding box constants
X_MIN, X_MAX = -90.0, 90.0      # latitude range
Y_MIN, Y_MAX = -176.3, 177.5    # longitude range


def duplicate_elimination(data_path, data_path_new):
    """
    Remove duplicate locations and invalid coordinates from the dataset.

    Input:
      - data_path (str):     path to original loc-gowalla_totalCheckins.txt
      - data_path_new (str): output path for cleaned dataset
    Output:
      Writes a tab-separated file: latitude\tlongitude\tlocation_id
    Behavior:
      - Extracts columns 3,4,5 (1-based) => indices 2,3,4 (0-based).
      - Removes points whose coordinates are outside the bounding box [X_MIN..X_MAX] (latitude)
        and [Y_MIN..Y_MAX] (longitude).
      - For points with identical (latitude, longitude) we keep the one with the smallest location id.
    """
    # Use a dict keyed by the textual (lat_str, lon_str) to preserve formatting; store smallest id
    coord_to_record = {}  # (lat_str, lon_str) -> (locid_int, lat_str, lon_str)

    with open(data_path, "r", encoding="utf-8", errors="ignore") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            # split on whitespace (handles tabs or spaces)
            parts = line.split()
            # we need at least 5 columns (1-based): keep indices 2,3,4
            if len(parts) < 5:
                continue
            lat_str = parts[2]
            lon_str = parts[3]
            locid_str = parts[4]
            try:
                lat = float(lat_str)
                lon = float(lon_str)
                locid = int(locid_str)
            except Exception:
                # malformed numbers -> skip
                continue

            # discard out of bounds
            if not (X_MIN <= lat <= X_MAX and Y_MIN <= lon <= Y_MAX):
                continue

            key = (lat_str, lon_str)
            if key in coord_to_record:
                existing_id = coord_to_record[key][0]
                if locid < existing_id:
                    coord_to_record[key] = (locid, lat_str, lon_str)
            else:
                coord_to_record[key] = (locid, lat_str, lon_str)

    # Write cleaned file: latitude\tlongitude\tlocation_id
    # Write in a deterministic order: sort by latitude (float) then longitude (float)
    items = []
    for (lat_s, lon_s), (locid, _, __) in coord_to_record.items():
        try:
            items.append((float(lat_s), float(lon_s), locid, lat_s, lon_s))
        except Exception:
            # fallback to lexical if float conversion fails
            items.append((0.0, 0.0, locid, lat_s, lon_s))

    items.sort(key=lambda x: (x[0], x[1], x[2]))

    with open(data_path_new, "w", encoding="utf-8") as fout:
        for lat_f, lon_f, locid, lat_s, lon_s in items:
            fout.write(f"{lat_s}\t{lon_s}\t{locid}\n")

    pass


def create_index(data_path_new, index_path, n):
    """
    Build an n*n grid index from the cleaned dataset and save to disk.

    Input:
      - data_path_new (str): path to cleaned dataset
      - index_path (str):    output path for the grid index file
      - n (int):             grid size (n x n cells)
    Output:
      Writes index file with format: Cell row, col: id_lat_lon id_lat_lon ...
    """
    if n <= 0:
        raise ValueError("n must be positive")

    lat_range = X_MAX - X_MIN
    lon_range = Y_MAX - Y_MIN
    cell_height = lat_range / n
    cell_width = lon_range / n
    EPS = 1e-12

    # Initialize empty lists for each cell
    cells = {}
    for r in range(n):
        for c in range(n):
            cells[(r, c)] = []

    def assign_cell(lat, lon):
        # compute t values
        # handle latitude (rows) with rule: boundary -> bottom cell (except if on X_MAX -> last row)
        # handle longitude (cols) with rule: boundary -> right cell (except if on Y_MAX -> last col)
        # t_lat = (lat - X_MIN) / cell_height ; t_lon = (lon - Y_MIN) / cell_width

        # Latitude -> row
        if abs(lat - X_MAX) <= EPS:
            row = 0
        else:
            t_lat = (lat - X_MIN) / cell_height  # position in cell-units measured from bottom
            t_lat_round = round(t_lat)
            if abs(t_lat - t_lat_round) <= EPS:
                # on a horizontal grid line: assign to the cell on its bottom (south)
                bottom_idx = int(t_lat_round) - 1
                if bottom_idx < 0:
                    # this means lat is on X_MIN (bottommost maxBox); assign to last row
                    row = n - 1
                else:
                    # convert bottom-index (0=bottom) to row-from-top
                    row = n - 1 - bottom_idx
            else:
                floor_t = int(math.floor(t_lat))
                row = n - 1 - floor_t


        # Longitude -> col
        if abs(lon - Y_MAX) <= EPS:
            col = n - 1
        else:
            t_lon = (lon - Y_MIN) / cell_width
            t_lon_round = round(t_lon)
            if abs(t_lon - t_lon_round) <= EPS:
                # boundary -> right cell
                col = int(t_lon_round)
            else:
                col = int(math.floor(t_lon))
        # clamp just in case
        if row < 0: row = 0
        if row >= n: row = n - 1
        if col < 0: col = 0
        if col >= n: col = n - 1

        return row, col

    # Read cleaned data and populate cells
    with open(data_path_new, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            lat_str = parts[0]
            lon_str = parts[1]
            locid_str = parts[2]
            try:
                lat = float(lat_str)
                lon = float(lon_str)
                locid = int(locid_str)
            except Exception:
                continue

            # assign to cell
            row, col = assign_cell(lat, lon)

            # store as "id_lat_lon" using original strings for lat/lon
            entry = f"{locid}_{lat_str}_{lon_str}"
            cells[(row, col)].append(entry)

    # Write index file in row-major order
    with open(index_path, "w", encoding="utf-8") as fout:
        for r in range(n):
            for c in range(n):
                entries = cells.get((r, c), [])
                if entries:
                    line = f"Cell {r}, {c}: " + " ".join(entries) + "\n"
                else:
                    line = f"Cell {r}, {c}:\n"
                fout.write(line)
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Part 1: Data preparation and grid index construction")
    parser.add_argument("data_path",
                        help="path to original Gowalla_totalCheckins.txt")
    parser.add_argument("index_path",
                        help="output path for the grid index file")
    parser.add_argument("data_path_new",
                        help="output path for deduplicated dataset")
    parser.add_argument("n", type=int,
                        help="grid size (n x n cells)")
    args = parser.parse_args()

    duplicate_elimination(args.data_path, args.data_path_new)

    s = time.time()
    create_index(args.data_path_new, args.index_path, args.n)
    t = time.time()
    print(f"Index construction time: {(t - s) * 1000:.2f} ms")
