# k-NN Search

This project implements a **grid-based spatial indexing** system for the Gowalla dataset, followed by **k-nearest neighbor (k-NN) search** algorithms.

---

## 1. Grid Index Construction

Build a grid index from the raw Gowalla check-in dataset.

### Usage

```bash
python3 make_index.py <input_file> <index_file> <cleaned_file> <n>
```

- `<input_file>`: raw Gowalla dataset (e.g., `loc-gowalla_totalCheckins.txt`)
- `<index_file>`: output index file (e.g., `index.txt`)
- `<cleaned_file>`: output deduplicated data file (e.g., `cleaned.txt`)
- `<n>`: number of grid cells per dimension (grid size = `n × n`)

### Example

```bash
python3 make_index.py loc-gowalla_totalCheckins.txt index.txt cleaned.txt 100
```

**Expected output** (if successful):

```
Index construction time: 1297.96 ms
```

---

## 2. k-Nearest Neighbor Search

Perform k-NN search using three methods:
- Linear scan
- Grid-based layer-by-layer search
- Grid-based best-first search

### Usage

```bash
python3 knn_search.py <q_x> <q_y> <cleaned_file> <index_file> <k> <n>
```

- `<q_x>`, `<q_y>`: query point coordinates
- `<cleaned_file>`: path to the deduplicated data file (from step 1)
- `<index_file>`: path to the grid index file (from step 1)
- `<k>`: number of nearest neighbors to find
- `<n>`: number of grid cells per dimension (must match the value used in index construction)

### Example

```bash
python3 knn_search.py 90.0 177.5 ./cleaned.txt ./index.txt 5 100
```

**Expected output** (if successful):

```
Linear scan results: 1434002, 1388864, 1388240, 2456633, 384920
Linear scan time: 803.01 ms
Grid (layer-by-layer) results: 1434002, 1388864, 1388240, 2456633, 384920
Grid (layer-by-layer) time: 616.67 ms, cells visited: 9
Grid (best-first) results: 1434002, 1388864, 1388240, 2456633, 384920
Grid (best-first) time: 589.85 ms, cells visited: 5
```

---

## Notes

- Ensure that the `<n>` parameter is consistent between index construction and search.
- File paths in the example are relative — adjust according to your directory structure.
