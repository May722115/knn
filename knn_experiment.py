#!/usr/bin/env python3
"""
k-NN experiments (loads on-disk index files into memory).
Usage example:
  python3 knn_experiments.py --run_experiments path/to/data_path_new --index_template "index_n{n}.txt" --out_prefix results
"""
import argparse
import math
import heapq
import time
import random
import os

# Bounding box constants
X_MIN, X_MAX = -90.0, 90.0      # latitude range
Y_MIN, Y_MAX = -176.3, 177.5    # longitude range
EPS = 1e-12

def make_grid_geometry(n):
    """Return (cell_lat_bounds, cell_lon_bounds, assign_cell) functions for grid size n."""
    lat_range = X_MAX - X_MIN
    lon_range = Y_MAX - Y_MIN
    cell_h = lat_range / n
    cell_w = lon_range / n

    def cell_lat_bounds(row):
        # row 0 = top; convert to bottom-based index
        bottom_index = n - 1 - row
        lat_min = X_MIN + bottom_index * cell_h
        lat_max = lat_min + cell_h
        return lat_min, lat_max

    def cell_lon_bounds(col):
        lon_min = Y_MIN + col * cell_w
        lon_max = lon_min + cell_w
        return lon_min, lon_max

    def assign_cell(lat, lon):
        # Latitude -> row
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

        # Longitude -> col
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
        if row < 0: row = 0
        if row >= n: row = n - 1
        if col < 0: col = 0
        if col >= n: col = n - 1
        return row, col

    return cell_lat_bounds, cell_lon_bounds, assign_cell


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

def load_dataset(data_path_new):
    """
    Load deduplicated dataset lines: latitude \t longitude \t location_id
    Returns list of (lat:float, lon:float, locid (int or str))
    """
    data = []
    with open(data_path_new, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                locid_raw = parts[2]
                try:
                    locid = int(locid_raw)
                except Exception:
                    locid = locid_raw
                data.append((lat, lon, locid))
            except Exception:
                continue
    return data

def load_index_from_file(index_path, n, verbose=False):
    """
    Load an on-disk index file in format:
       Cell r, c: id_lat_lon id_lat_lon ...
    Returns grid: list of lists grid[r][c] -> list of (locid, lat, lon)
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found: {index_path}")
    grid = [[[] for _ in range(n)] for __ in range(n)]
    total_pts = 0
    nonempty = 0
    with open(index_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            if not line.startswith("Cell"):
                continue
            try:
                header, rest = line.split(":", 1)
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
            for token in tokens:
                # token expected: id_lat_lon (lat and lon may be negative, keep split to max 3)
                parts = token.split("_", 2)
                if len(parts) != 3:
                    continue
                locid_s, lat_s, lon_s = parts
                try:
                    locid = int(locid_s)
                except Exception:
                    locid = locid_s
                try:
                    lat = float(lat_s)
                    lon = float(lon_s)
                except Exception:
                    continue
                if 0 <= r < n and 0 <= c < n:
                    grid[r][c].append((locid, lat, lon))
                    total_pts += 1
    # count nonempty
    for r in range(n):
        for c in range(n):
            if grid[r][c]:
                nonempty += 1
    if verbose:
        print(f"Loaded index {index_path}: total_points={total_pts}, nonempty_cells={nonempty}/{n*n}")
    return grid

def knn_linear_inmemory(x, y, data_list, k):
    if k <= 0:
        return ""
    heap = []
    for (plat, plon, locid) in data_list:
        dx = plat - x
        dy = plon - y
        dist_sq = dx * dx + dy * dy
        if len(heap) < k:
            heapq.heappush(heap, (-dist_sq, locid))
        else:
            if dist_sq < -heap[0][0]:
                heapq.heapreplace(heap, (-dist_sq, locid))
    res = []
    while heap:
        negd, lid = heapq.heappop(heap)
        res.append((-negd, lid))
    res.sort(key=lambda item: item[0])
    ids = [str(item[1]) for item in res]
    return ", ".join(ids)

def knn_grid_inmemory(x, y, grid, k, n):
    """Layer-by-layer grid search. Returns (ids_str, cells_visited)"""
    if k <= 0:
        return "", 0
    cell_lat_bounds, cell_lon_bounds, assign_cell = make_grid_geometry(n)
    r0, c0 = assign_cell(x, y)
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
        # current threshold t_sq (squared) from heap
        if len(heap) < k:
            t_sq = float("inf")
        else:
            t_sq = -heap[0][0]
        # compute minimal dlow for the layer
        layer_min = float("inf")
        cell_dlow = []
        for (rr, cc) in cells_in_layer:
            lat_min, lat_max = cell_lat_bounds(rr)
            lon_min, lon_max = cell_lon_bounds(cc)
            dlow_sq = dlow_sq_for_cell(x, y, lat_min, lat_max, lon_min, lon_max)
            cell_dlow.append((rr, cc, dlow_sq))
            if dlow_sq < layer_min:
                layer_min = dlow_sq
        if layer_min > t_sq:
            break
        accessed_any = False
        for (rr, cc, dlow_sq) in cell_dlow:
            if len(heap) < k:
                t_sq = float("inf")
            else:
                t_sq = -heap[0][0]
            if dlow_sq > t_sq:
                continue
            pts = grid[rr][cc]
            if pts:
                accessed_any = True
                cells_visited += 1
                for (locid, plat, plon) in pts:
                    dx = plat - x
                    dy = plon - y
                    dist_sq = dx * dx + dy * dy
                    if len(heap) < k:
                        heapq.heappush(heap, (-dist_sq, locid))
                    else:
                        if dist_sq < -heap[0][0]:
                            heapq.heapreplace(heap, (-dist_sq, locid))
        # small optimization: if layer had no cells accessed and layer > 0, and layer_min is 0,
        # we still continue to next layer; we only break above when layer_min > t_sq
    res = []
    while heap:
        negd, lid = heapq.heappop(heap)
        res.append((-negd, lid))
    res.sort(key=lambda item: item[0])
    ids = [str(item[1]) for item in res]
    return ", ".join(ids), cells_visited

def knn_grid_bf_inmemory(x, y, grid, k, n):
    """Best-first grid search. Returns (ids_str, cells_visited)"""
    if k <= 0:
        return "", 0
    cell_lat_bounds, cell_lon_bounds, assign_cell = make_grid_geometry(n)
    r0, c0 = assign_cell(x, y)
    visited = [[False] * n for _ in range(n)]
    cell_pq = []
    lat_min0, lat_max0 = cell_lat_bounds(r0)
    lon_min0, lon_max0 = cell_lon_bounds(c0)
    d0 = dlow_sq_for_cell(x, y, lat_min0, lat_max0, lon_min0, lon_max0)
    heapq.heappush(cell_pq, (d0, r0, c0))
    heap = []
    cells_visited = 0
    while cell_pq:
        dlow_sq, rr, cc = heapq.heappop(cell_pq)
        if visited[rr][cc]:
            continue
        if len(heap) < k:
            t_sq = float("inf")
        else:
            t_sq = -heap[0][0]
        if dlow_sq > t_sq:
            break
        visited[rr][cc] = True
        pts = grid[rr][cc]
        if pts:
            cells_visited += 1
            for (locid, plat, plon) in pts:
                dx = plat - x
                dy = plon - y
                dist_sq = dx * dx + dy * dy
                if len(heap) < k:
                    heapq.heappush(heap, (-dist_sq, locid))
                else:
                    if dist_sq < -heap[0][0]:
                        heapq.heapreplace(heap, (-dist_sq, locid))
        # push neighbors
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
                ndlow_sq = dlow_sq_for_cell(x, y, lat_min, lat_max, lon_min, lon_max)
                heapq.heappush(cell_pq, (ndlow_sq, ni, nj))
    res = []
    while heap:
        negd, lid = heapq.heappop(heap)
        res.append((-negd, lid))
    res.sort(key=lambda item: item[0])
    ids = [str(item[1]) for item in res]
    return ", ".join(ids), cells_visited

# ----------R-tree implementation (STR bulk-load + best-first kNN)----------

class RTreeNode:
    def __init__(self, is_leaf):
        self.is_leaf = is_leaf
        self.children = []  # for leaf: list of points(locid, lat, lon); for non-leaf: list of child nodes
        self.mbr = None      # (minx, miny, maxx, maxy)

def mbr_of_point(lat, lon):
    return (lat, lon, lat, lon)

def extend_mbr(mbr, other):
    if mbr is None:
        return other
    minx = min(mbr[0], other[0])
    miny = min(mbr[1], other[1])
    maxx = max(mbr[2], other[2])
    maxy = max(mbr[3], other[3])
    return (minx, miny, maxx, maxy)

def mbr_of_node(node):
    m = None
    if node.is_leaf:
        for (locid, lat, lon) in node.children:
            m = extend_mbr(m, mbr_of_point(lat, lon))
    else:
        for child in node.children:
            m = extend_mbr(m, child.mbr)
    node.mbr = m
    return m

def centroid_of_node(node):
    # centroid of MBR
    minx, miny, maxx, maxy = node.mbr
    return ((minx + maxx) / 2.0, (miny + maxy) / 2.0)

def build_rtree_str(data_list, leaf_capacity=32, node_capacity=32):
    """
    Build an R-tree using the STR (Sort-Tile-Recursive) bulk-load algorithm.
    data_list: list of (lat, lon, locid)
    leaf_capacity, node_capacity: max entries per node
    Returns root node.
    """
    if not data_list:
        root = RTreeNode(is_leaf=True)
        root.mbr = None
        return root

    N = len(data_list)
    # convert to (x=lat, y=lon, locid)
    points = [(lat, lon, locid) for (lat, lon, locid) in data_list]

    # Create leaf nodes:
    L = leaf_capacity
    S = int(math.ceil(N / L))                      # target number of leaf nodes
    slice_count = int(math.ceil(math.sqrt(S)))     # number of vertical slices
    if slice_count < 1:
        slice_count = 1
    points_sorted_x = sorted(points, key=lambda p: p[0])  # sort by x (lat)

    leaves = []
    slice_size = int(math.ceil(len(points_sorted_x) / slice_count))
    idx = 0
    while idx < len(points_sorted_x):
        slice_block = points_sorted_x[idx: idx + slice_size]
        idx += slice_size
        # sort slice by y and pack into leaf nodes
        slice_sorted_y = sorted(slice_block, key=lambda p: p[1])
        j = 0
        while j < len(slice_sorted_y):
            group = slice_sorted_y[j: j + L]
            j += L
            leaf = RTreeNode(is_leaf=True)
            # store as (locid, lat, lon) for consistency with grid leaf format
            leaf.children = [(item[2], item[0], item[1]) for item in group]
            mbr_of_node(leaf)
            leaves.append(leaf)

    # Build upper levels by packing children into nodes of capacity node_capacity
    current_level = leaves
    while len(current_level) > 1:
        # sort nodes by centroid.x (lat)
        current_level_sorted = sorted(current_level, key=lambda node: centroid_of_node(node)[0])
        new_level = []
        group_size = node_capacity
        i = 0
        while i < len(current_level_sorted):
            group = current_level_sorted[i: i + group_size]
            i += group_size
            parent = RTreeNode(is_leaf=False)
            parent.children = group
            mbr_of_node(parent)
            new_level.append(parent)
        current_level = new_level

    root = current_level[0]
    return root

def generate_queries(num_queries=100, seed=42, out_file=None):
    random.seed(seed)
    qs = []
    for _ in range(num_queries):
        lat = random.uniform(X_MIN, X_MAX)
        lon = random.uniform(Y_MIN, Y_MAX)
        qs.append((lat, lon))
    if out_file:
        with open(out_file, "w", encoding="utf-8") as fout:
            for (lat, lon) in qs:
                fout.write(f"{lat} {lon}\n")
    return qs

def load_queries_from_file(queries_path):
    qs = []
    with open(queries_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                lat = float(parts[0]); lon = float(parts[1])
                qs.append((lat, lon))
            except Exception:
                continue
    return qs

def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0

def run_experiments(data_path_new, queries_file=None, out_prefix=None, index_template=None):
    print("Loading dataset (for linear scan) ...")
    data_list = load_dataset(data_path_new)
    print(f"Loaded {len(data_list)} points from {data_path_new}.")

    if queries_file:
        queries = load_queries_from_file(queries_file)
        print(f"Loaded {len(queries)} queries from {queries_file}.")
    else:
        queries = generate_queries(100, seed=42)
        print(f"Generated {len(queries)} deterministic queries (seed=42).")

    # Build R-tree once
    print("Building R-tree (STR bulk-load) ...", end="", flush=True)
    # leaf/node capacities can be tuned
    rtree_root = build_rtree_str(data_list, leaf_capacity=32, node_capacity=32)
    print(" done.")

    ns = [10, 50, 100, 150, 200]
    k_fixed = 5
    exp1_results = []

    print("\nExperiment 1: Varying grid size n (k fixed = 5)")
    for n in ns:
        if not index_template:
            raise ValueError("index_template must be provided to load pre-built index files.")
        index_path = index_template.format(n=n)
        print(f" Loading index file for n={n}: {index_path} ...", end="", flush=True)
        grid = load_index_from_file(index_path, n, verbose=False)
        print(" done.")
        linear_times = []
        grid_times = []
        grid_cells = []
        bf_times = []
        bf_cells = []
        rtree_times = []
        rtree_nodes = []
        rtree_points = []

        for (qx, qy) in queries:
            t0 = time.perf_counter()
            _ = knn_linear_inmemory(qx, qy, data_list, k_fixed)
            t1 = time.perf_counter()
            linear_times.append((t1 - t0) * 1000.0)

            t0 = time.perf_counter()
            _, cells_g = knn_grid_inmemory(qx, qy, grid, k_fixed, n)
            t1 = time.perf_counter()
            grid_times.append((t1 - t0) * 1000.0)
            grid_cells.append(cells_g)

            t0 = time.perf_counter()
            _, cells_bf = knn_grid_bf_inmemory(qx, qy, grid, k_fixed, n)
            t1 = time.perf_counter()
            bf_times.append((t1 - t0) * 1000.0)
            bf_cells.append(cells_bf)

            # R-tree best-first
            t0 = time.perf_counter()
            _, nodes_vis, pts_vis = knn_rtree_inmemory(qx, qy, rtree_root, k_fixed)
            t1 = time.perf_counter()
            rtree_times.append((t1 - t0) * 1000.0)
            rtree_nodes.append(nodes_vis)
            rtree_points.append(pts_vis)

        row = {
            "n": n,
            "linear_ms": avg(linear_times),
            "grid_ms": avg(grid_times),
            "grid_cells": avg(grid_cells),
            "bf_ms": avg(bf_times),
            "bf_cells": avg(bf_cells),
            "rtree_ms": avg(rtree_times),
            "rtree_nodes": avg(rtree_nodes),
            "rtree_points": avg(rtree_points),
        }
        exp1_results.append(row)
        print(f" n={n} done: linear {row['linear_ms']:.3f} ms, grid {row['grid_ms']:.3f} ms ({row['grid_cells']:.2f} cells), bf {row['bf_ms']:.3f} ms ({row['bf_cells']:.2f} cells), rtree {row['rtree_ms']:.3f} ms")

    print("\nExperiment 1 results (avg over queries):")
    print("n, linear_ms, knn_grid_ms, knn_grid_cells_avg, knn_grid_bf_ms, knn_grid_bf_cells_avg, knn_rtree_ms, knn_rtree_nodes_avg, knn_rtree_points_avg")
    for r in exp1_results:
        print(f"{r['n']}, {r['linear_ms']:.6f}, {r['grid_ms']:.6f}, {r['grid_cells']:.6f}, {r['bf_ms']:.6f}, {r['bf_cells']:.6f}, {r['rtree_ms']:.6f}, {r['rtree_nodes']:.6f}, {r['rtree_points']:.6f}")

    if out_prefix:
        out1 = f"{out_prefix}_exp1.csv"
        with open(out1, "w", encoding="utf-8") as fout:
            fout.write("n,linear_ms,knn_grid_ms,knn_grid_cells_avg,knn_grid_bf_ms,knn_grid_bf_cells_avg,knn_rtree_ms,knn_rtree_nodes_avg,knn_rtree_points_avg\n")
            for r in exp1_results:
                fout.write(f"{r['n']},{r['linear_ms']},{r['grid_ms']},{r['grid_cells']},{r['bf_ms']},{r['bf_cells']},{r['rtree_ms']},{r['rtree_nodes']},{r['rtree_points']}\n")
        print(f"Saved Experiment 1 results to {out1}")

    # Experiment 2: vary k for fixed n
    n_fixed = 50
    ks = list(range(1, 11))
    print(f"\nExperiment 2: Varying k (n fixed = {n_fixed})")
    index_path = index_template.format(n=n_fixed)
    print(f" Loading index for n={n_fixed}: {index_path} ...", end="", flush=True)
    grid = load_index_from_file(index_path, n_fixed, verbose=False)
    print(" done.")

    exp2_results = []
    for k in ks:
        linear_times = []
        grid_times = []
        grid_cells = []
        bf_times = []
        bf_cells = []
        rtree_times = []
        rtree_nodes = []
        rtree_points = []
        for (qx, qy) in queries:
            t0 = time.perf_counter()
            _ = knn_linear_inmemory(qx, qy, data_list, k)
            t1 = time.perf_counter()
            linear_times.append((t1 - t0) * 1000.0)

            t0 = time.perf_counter()
            _, cells_g = knn_grid_inmemory(qx, qy, grid, k, n_fixed)
            t1 = time.perf_counter()
            grid_times.append((t1 - t0) * 1000.0)
            grid_cells.append(cells_g)

            t0 = time.perf_counter()
            _, cells_bf = knn_grid_bf_inmemory(qx, qy, grid, k, n_fixed)
            t1 = time.perf_counter()
            bf_times.append((t1 - t0) * 1000.0)
            bf_cells.append(cells_bf)

            # R-tree
            t0 = time.perf_counter()
            _, nodes_vis, pts_vis = knn_rtree_inmemory(qx, qy, rtree_root, k)
            t1 = time.perf_counter()
            rtree_times.append((t1 - t0) * 1000.0)
            rtree_nodes.append(nodes_vis)
            rtree_points.append(pts_vis)


        row = {
            "k": k,
            "linear_ms": avg(linear_times),
            "grid_ms": avg(grid_times),
            "grid_cells": avg(grid_cells),
            "bf_ms": avg(bf_times),
            "bf_cells": avg(bf_cells),
            "rtree_ms": avg(rtree_times),
            "rtree_nodes": avg(rtree_nodes),
            "rtree_points": avg(rtree_points),
        }
        exp2_results.append(row)
        print(f" k={k} done: linear {row['linear_ms']:.3f} ms, grid {row['grid_ms']:.3f} ms ({row['grid_cells']:.2f} cells), bf {row['bf_ms']:.3f} ms ({row['bf_cells']:.2f} cells), rtree {row['rtree_ms']:.3f} ms")

    print("\nExperiment 2 results (avg over queries):")
    print("k, linear_ms, knn_grid_ms, knn_grid_cells_avg, knn_grid_bf_ms, knn_grid_bf_cells_avg, knn_rtree_ms, knn_rtree_nodes_avg, knn_rtree_points_avg")
    for r in exp2_results:
        print(f"{r['k']}, {r['linear_ms']:.6f}, {r['grid_ms']:.6f}, {r['grid_cells']:.6f}, {r['bf_ms']:.6f}, {r['bf_cells']:.6f}, {r['rtree_ms']:.6f}, {r['rtree_nodes']:.6f}, {r['rtree_points']:.6f}")

    if out_prefix:
        out2 = f"{out_prefix}_exp2.csv"
        with open(out2, "w", encoding="utf-8") as fout:
            fout.write("k,linear_ms,knn_grid_ms,knn_grid_cells_avg,knn_grid_bf_ms,knn_grid_bf_cells_avg,knn_rtree_ms,knn_rtree_nodes_avg,knn_rtree_points_avg\n")
            for r in exp2_results:
                fout.write(f"{r['k']},{r['linear_ms']},{r['grid_ms']},{r['grid_cells']},{r['bf_ms']},{r['bf_cells']},{r['rtree_ms']},{r['rtree_nodes']},{r['rtree_points']}\n")
        print(f"Saved Experiment 2 results to {out2}")

    print("\nExperiments complete.")
    return exp1_results, exp2_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run k-NN experiments loading pre-built index files.")
    parser.add_argument("data_path_new", help="path to deduplicated dataset (used for linear scan)")
    parser.add_argument("--run_experiments", action="store_true",
                        help="Run Experiments 1 and 2 (loads index files via --index_template).")
    parser.add_argument("--index_template", default=None,
                        help='Index filename template with "{n}" placeholder (e.g., "index_n{n}.txt"). Required for --run_experiments.')
    parser.add_argument("--queries", default=None, help="Optional queries file (overrides generated queries).")
    parser.add_argument("--out_prefix", default=None, help="Optional output CSV prefix for experiment results.")
    args = parser.parse_args()

    if args.run_experiments:
        if not args.index_template:
            print("ERROR: --index_template is required when using --run_experiments (e.g. \"index_n{n}.txt\").")
            raise SystemExit(1)
        try:
            run_experiments(args.data_path_new, queries_file=args.queries, out_prefix=args.out_prefix, index_template=args.index_template)
        except Exception as e:
            print("Error during experiments:", e)
            raise
    else:
        parser.print_help()
