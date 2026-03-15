#!/usr/bin/env python3
"""
COMP3323 Assignment 2 - Parts 2, 3 & 4(1): k-NN Search Algorithms
  - knn_linear_scan  (Part 4.1)
  - knn_grid         (Part 2: layer-by-layer expansion)
  - knn_grid_bf      (Part 3: best-first cell expansion)
"""

import sys
import time
import math
import heapq
import argparse
import os

# Bounding box constants
X_MIN, X_MAX = -90.0, 90.0
Y_MIN, Y_MAX = -176.3, 177.5
EPS = 1e-12


# ---------------------------------------------------------------------------
# Helper functions 
# ---------------------------------------------------------------------------

def dist_sq(ax, ay, bx, by):
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy

def load_dataset(data_path):
    """
    Load deduplicated dataset. Expect lines: lat lon id
    Returns list of tuples: (lat, lon, id_int)
    """
    pts = []
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                locid = int(parts[2])
                pts.append((lat, lon, locid))
            except Exception:
                # skip malformed lines
                continue
    return pts

def make_grid_geometry(n):
    """
    Returns (cell_lat_bounds, cell_lon_bounds, assign_cell)
    - cell_lat_bounds(row) -> (lat_min, lat_max)
    - cell_lon_bounds(col) -> (lon_min, lon_max)
    - assign_cell(lat, lon) -> (row, col)
    Row 0 corresponds to top (highest lat), row n-1 to bottom (lowest lat).
    """
    lat_range = X_MAX - X_MIN
    lon_range = Y_MAX - Y_MIN
    cell_h = lat_range / n
    cell_w = lon_range / n

    def cell_lat_bounds(row):
        # convert row (0 = top) to bottom-indexed
        bottom_index = n - 1 - row
        lat_min = X_MIN + bottom_index * cell_h
        lat_max = lat_min + cell_h
        return lat_min, lat_max

    def cell_lon_bounds(col):
        lon_min = Y_MIN + col * cell_w
        lon_max = lon_min + cell_w
        return lon_min, lon_max

    def assign_cell(lat, lon):
        # row
        if abs(lat - X_MAX) <= EPS:
            row = 0
        else:
            t_lat = (lat - X_MIN) / cell_h
            t_lat_round = round(t_lat)
            if abs(t_lat - t_lat_round) <= EPS:
                bottom_idx = int(t_lat_round) - 1
                if bottom_idx < 0:
                    row = n - 1
                else:
                    row = n - 1 - bottom_idx
            else:
                floor_t = int(math.floor(t_lat))
                row = n - 1 - floor_t
        # col
        if abs(lon - Y_MAX) <= EPS:
            col = n - 1
        else:
            t_lon = (lon - Y_MIN) / cell_w
            t_lon_round = round(t_lon)
            if abs(t_lon - t_lon_round) <= EPS:
                col = int(t_lon_round)
            else:
                col = int(math.floor(t_lon))
        # clamp
        row = max(0, min(n - 1, row))
        col = max(0, min(n - 1, col))
        return row, col

    return cell_lat_bounds, cell_lon_bounds, assign_cell

def load_index_from_file(path, n):
    """
    Load grid index file formatted as:
        Cell r, c: id_lat_lon id_lat_lon ...
    where id, latitude, longitude are separated by underscores.
    Returns grid[r][c] -> list of (locid_int, lat, lon)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index file not found: {path}")
    grid = [[[] for _ in range(n)] for __ in range(n)]
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or not s.startswith("Cell"):
                continue
            try:
                header, rest = s.split(":", 1)
                rc = header[len("Cell"):].strip()
                r_str, c_str = rc.split(",")
                r = int(r_str.strip())
                c = int(c_str.strip())
            except Exception:
                continue
            rest = rest.strip()
            if not rest:
                continue
            tokens = rest.split()
            for tok in tokens:
                parts = tok.split("_", 2)
                if len(parts) != 3:
                    continue
                locid_s, lat_s, lon_s = parts
                try:
                    lat = float(lat_s)
                    lon = float(lon_s)
                    locid = int(locid_s)
                except Exception:
                    continue
                if 0 <= r < n and 0 <= c < n:
                    grid[r][c].append((locid, lat, lon))
    return grid

def dlow_sq_for_cell(lat_q, lon_q, lat_min, lat_max, lon_min, lon_max):
    dx = 0.0
    if lat_q < lat_min:
        dx = lat_min - lat_q
    elif lat_q > lat_max:
        dx = lat_q - lat_max
    dy = 0.0
    if lon_q < lon_min:
        dy = lon_min - lon_q
    elif lon_q > lon_max:
        dy = lon_q - lon_max
    return dx * dx + dy * dy

def format_result_ids(id_list):
    return ", ".join(str(i) for i in id_list)

# ---------------------------------------------------------------------------
# Part 2: Grid k-NN (layer-by-layer expansion)
# ---------------------------------------------------------------------------
def knn_grid(x, y, index_path, k, n):
    """
    Find k nearest neighbors using grid index with layer-by-layer expansion.

    Input:
      - x (float):          latitude of query point
      - y (float):          longitude of query point
      - index_path (str):   path to the grid index file
      - k (int):            number of nearest neighbors
      - n (int):            grid size (n x n)
    Output:
      - tuple: (result_str, cells_visited)
        - result_str (str): comma-separated location ids, e.g. "11, 789, 125, 2, 771"
        - cells_visited (int): number of cells whose points were examined
    """
    # load index
    grid = load_index_from_file(index_path, n)

    cell_lat_bounds, cell_lon_bounds, assign_cell = make_grid_geometry(n)
    r0, c0 = assign_cell(x, y)

    # max-heap implemented via min-heap with tuples (-dist, -locid)
    heap = []
    cells_visited = 0

    max_layer = max(r0, c0, n - 1 - r0, n - 1 - c0)
    for layer in range(0, max_layer + 1):
        cells_in_layer = []
        for dr in range(-layer, layer + 1):
            for dc in range(-layer, layer + 1):
                if max(abs(dr), abs(dc)) != layer:
                    continue
                rr = r0 + dr
                cc = c0 + dc
                if 0 <= rr < n and 0 <= cc < n:
                    cells_in_layer.append((rr, cc))
        # compute minimal dlow for the whole layer
        layer_min = float("inf")
        cell_dlows = []
        for (rr, cc) in cells_in_layer:
            lat_min, lat_max = cell_lat_bounds(rr)
            lon_min, lon_max = cell_lon_bounds(cc)
            dlow = dlow_sq_for_cell(x, y, lat_min, lat_max, lon_min, lon_max)
            cell_dlows.append((rr, cc, dlow))
            if dlow < layer_min:
                layer_min = dlow

        if len(heap) < k:
            t_sq = float("inf")
        else:
            # worst distance in heap is -heap[0][0]
            t_sq = -heap[0][0]

        # if the entire layer's minimal possible distance > current worst, we can stop
        if layer_min > t_sq:
            break

        # examine cells in this layer
        for (rr, cc, dlow) in cell_dlows:
            if len(heap) < k:
                t_sq = float("inf")
            else:
                t_sq = -heap[0][0]
            if dlow > t_sq:
                # this cell cannot contain closer points
                continue
            pts = grid[rr][cc]
            if pts:
                cells_visited += 1
            for (locid, plat, plon) in pts:
                d = dist_sq(x, y, plat, plon)
                tup = (-d, -locid)
                if len(heap) < k:
                    heapq.heappush(heap, tup)
                else:
                    # replace if candidate is better than current worst
                    if tup > heap[0]:
                        heapq.heapreplace(heap, tup)

    # extract heap contents sorted ascending by (dist, id)
    res = []
    while heap:
        nd, nid = heapq.heappop(heap)
        res.append((-nd, -nid))
    # sort by distance then id
    res.sort(key=lambda item: (item[0], item[1]))
    ids = [locid for (_, locid) in res]
    return format_result_ids(ids), cells_visited

# ---------------------------------------------------------------------------
# Part 3: Grid k-NN (best-first cell expansion)
# ---------------------------------------------------------------------------
def knn_grid_bf(x, y, index_path, k, n):
    """
    Find k nearest neighbors using grid index with best-first cell expansion.

    Input:
      - x (float):          latitude of query point
      - y (float):          longitude of query point
      - index_path (str):   path to the grid index file
      - k (int):            number of nearest neighbors
      - n (int):            grid size (n x n)
    Output:
      - tuple: (result_str, cells_visited)
        - result_str (str): comma-separated location ids, e.g. "11, 789, 125, 2, 771"
        - cells_visited (int): number of cells whose points were examined
    """
    grid = load_index_from_file(index_path, n)
    cell_lat_bounds, cell_lon_bounds, assign_cell = make_grid_geometry(n)
    r0, c0 = assign_cell(x, y)

    visited = [[False] * n for _ in range(n)]
    cell_pq = []  # min-heap of (dlow_sq, r, c)

    lat_min0, lat_max0 = cell_lat_bounds(r0)
    lon_min0, lon_max0 = cell_lon_bounds(c0)
    d0 = dlow_sq_for_cell(x, y, lat_min0, lat_max0, lon_min0, lon_max0)
    heapq.heappush(cell_pq, (d0, r0, c0))

    heap = []  # max-heap for points via (-dist, -locid)
    cells_visited = 0

    while cell_pq:
        dlow_sq, rr, cc = heapq.heappop(cell_pq)
        if visited[rr][cc]:
            continue

        if len(heap) < k:
            t_sq = float("inf")
        else:
            t_sq = -heap[0][0]

        # if the best possible remaining cell cannot improve current worst -> stop
        if dlow_sq > t_sq:
            break

        # mark visited and examine points in cell
        visited[rr][cc] = True
        pts = grid[rr][cc]
        if pts:
            cells_visited += 1
        for (locid, plat, plon) in pts:
            d = dist_sq(x, y, plat, plon)
            tup = (-d, -locid)
            if len(heap) < k:
                heapq.heappush(heap, tup)
            else:
                if tup > heap[0]:
                    heapq.heapreplace(heap, tup)

        # push neighbors (8-connected) if not visited
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni = rr + di
                nj = cc + dj
                if ni < 0 or ni >= n or nj < 0 or nj >= n:
                    continue
                if visited[ni][nj]:
                    continue
                lat_min, lat_max = cell_lat_bounds(ni)
                lon_min, lon_max = cell_lon_bounds(nj)
                ndlow = dlow_sq_for_cell(x, y, lat_min, lat_max, lon_min, lon_max)
                heapq.heappush(cell_pq, (ndlow, ni, nj))

    # extract heap contents sorted ascending by (dist, id)
    res = []
    while heap:
        nd, nid = heapq.heappop(heap)
        res.append((-nd, -nid))
    res.sort(key=lambda item: (item[0], item[1]))
    ids = [locid for (_, locid) in res]
    return format_result_ids(ids), cells_visited

# ---------------------------------------------------------------------------
# Part 4(1): Linear scan
# ---------------------------------------------------------------------------
def knn_linear_scan(x, y, data_path_new, k):
    """
    Find k nearest neighbors by scanning all points.

    Input:
      - x (float):            latitude of query point
      - y (float):            longitude of query point
      - data_path_new (str):  path to the deduplicated dataset
      - k (int):              number of nearest neighbors
    Output:
      - tuple: (result_str, cells_visited)
        - result_str (str): comma-separated location ids, e.g. "11, 789, 125, 2, 771"
        - cells_visited (int): 0 (linear scan does not use grid cells)
    """
    pts = load_dataset(data_path_new)
    heap = []  # max-heap via (-dist, -locid)
    for (plat, plon, locid) in pts:
        d = dist_sq(x, y, plat, plon)
        tup = (-d, -locid)
        if len(heap) < k:
            heapq.heappush(heap, tup)
        else:
            if tup > heap[0]:
                heapq.heapreplace(heap, tup)
    # extract and sort
    res = []
    while heap:
        nd, nid = heapq.heappop(heap)
        res.append((-nd, -nid))
    res.sort(key=lambda item: (item[0], item[1]))
    ids = [locid for (_, locid) in res]
    return format_result_ids(ids), 0
  
# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parts 2 & 3: Run k-NN search using linear scan, "
                    "grid (layer-by-layer), and grid (best-first)")
    parser.add_argument("x", type=float,
                        help="latitude of the query point q")
    parser.add_argument("y", type=float,
                        help="longitude of the query point q")
    parser.add_argument("data_path_new",
                        help="path to the deduplicated dataset")
    parser.add_argument("index_path",
                        help="path to the grid index file")
    parser.add_argument("k", type=int,
                        help="number of nearest neighbors")
    parser.add_argument("n", type=int,
                        help="grid size (n x n cells)")
    args = parser.parse_args()

    # Linear scan
    s = time.time()
    result, _ = knn_linear_scan(args.x, args.y, args.data_path_new, args.k)
    t = time.time()
    print(f"Linear scan results: {result}")
    print(f"Linear scan time: {(t - s) * 1000:.2f} ms")

    # Grid (layer-by-layer)
    s = time.time()
    result, cells = knn_grid(args.x, args.y, args.index_path, args.k, args.n)
    t = time.time()
    print(f"Grid (layer-by-layer) results: {result}")
    print(f"Grid (layer-by-layer) time: {(t - s) * 1000:.2f} ms, cells visited: {cells}")

    # Grid (best-first)
    s = time.time()
    result, cells = knn_grid_bf(args.x, args.y, args.index_path, args.k, args.n)
    t = time.time()
    print(f"Grid (best-first) results: {result}")
    print(f"Grid (best-first) time: {(t - s) * 1000:.2f} ms, cells visited: {cells}")
