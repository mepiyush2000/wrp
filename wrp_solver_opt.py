import heapq
import itertools
import math
from collections import deque
import numpy as np
import torch

class WRPSolverTSPJF:
    # Set default vision_radius to infinity for standard WRP
    def __init__(self, grid, start, los_type='los4', vision_radius=float('inf'), use_gpu=False):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.start = start
        self.los_type = los_type.lower()
        self.vision_radius = vision_radius
        self.vision_radius_is_finite = math.isfinite(vision_radius)
        self.vision_radius_sq = vision_radius * vision_radius if self.vision_radius_is_finite else None
        self.empty_cells = set()
        
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == 0:
                    self.empty_cells.add((r, c))
                    
        self.los_table = {}
        self.los_mask_table = {}
        self.los_size_table = {}
        self.apsp_matrix = None  # [n, n] numpy array, set by _precompute_apsp

        # Device: MPS (Apple Silicon) > CUDA > CPU
        # Note: for typical grid sizes (<700 cells), CPU parallel BFS is fastest.
        # MPS becomes beneficial only for very large grids (n > ~1500).
        if use_gpu and torch.backends.mps.is_available():
            self._apsp_device = torch.device('mps')
        elif use_gpu and torch.cuda.is_available():
            self._apsp_device = torch.device('cuda')
        else:
            self._apsp_device = torch.device('cpu')

        self.empty_cells_list = sorted(self.empty_cells)
        self.cell_to_idx = {cell: idx for idx, cell in enumerate(self.empty_cells_list)}
        self.all_seen_mask = (1 << len(self.empty_cells_list)) - 1
        self.n_cells = len(self.empty_cells_list)
        
        # Determine vision directions for raycasting (if not using Bresenham)
        if self.los_type == 'los8':
            self.vision_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), 
                                (1, 1), (1, -1), (-1, 1), (-1, -1)]
        elif self.los_type == 'los4':
            self.vision_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            
        self._precompute_los()
        self._build_los_masks()
        self._precompute_apsp()
        # Caches populated during solve — keyed by (current_loc, unseen_mask) / unseen_mask
        self._heuristic_cache = {}
        self._pivots_cache = {}
        self._pivot_dist_cache = {}   # pivot-to-pivot submatrix per unseen_mask

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def get_neighbors(self, loc):
        r, c = loc
        neighbors = []
        # Even with Bresenham vision, movement is typically restricted to 4 or 8 ways.
        # Defaulting to 4-way movement here. Add diagonals if your agent can move diagonally.
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc) and self.grid[nr][nc] == 0:
                neighbors.append((nr, nc))
        return neighbors

    def _bresenham(self, r0, c0, r1, c1):
        """Yields coordinates on the straight line between (r0, c0) and (r1, c1)."""
        dy = abs(r1 - r0)
        dx = abs(c1 - c0)
        sy = 1 if r0 < r1 else -1
        sx = 1 if c0 < c1 else -1
        err = dx - dy

        while True:
            yield (r0, c0)
            if r0 == r1 and c0 == c1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                c0 += sx
            if e2 < dx:
                err += dx
                r0 += sy

    def _build_los_masks(self):
        for cell, visible_cells in self.los_table.items():
            mask = 0
            for v in visible_cells:
                mask |= 1 << self.cell_to_idx[v]
            self.los_mask_table[cell] = mask
            self.los_size_table[cell] = mask.bit_count()

        # Precompute numpy index arrays once — eliminates np.fromiter inside min_comp_dist
        self.los_idx_table = {
            cell: np.fromiter(self._iter_mask_indices(mask), dtype=np.intp)
            for cell, mask in self.los_mask_table.items()
        }

    def _iter_mask_indices(self, mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb

    def _iter_mask_cells(self, mask):
        for idx in self._iter_mask_indices(mask):
            yield self.empty_cells_list[idx]

    def _precompute_los(self):
        """Precomputes the LOS for every empty cell with an optional radius limit."""
        if self.los_type == 'bresenham':
            for cell in self.empty_cells:
                visible = set()
                r0, c0 = cell

                if self.vision_radius_is_finite:
                    radius_int = int(self.vision_radius)
                    r_min = max(0, r0 - radius_int)
                    r_max = min(self.rows, r0 + radius_int + 1)
                    c_min = max(0, c0 - radius_int)
                    c_max = min(self.cols, c0 + radius_int + 1)
                else:
                    r_min, r_max, c_min, c_max = 0, self.rows, 0, self.cols
                
                # We MUST iterate over the bounding box, casting rays to WALLS too
                for r1 in range(r_min, r_max):
                    for c1 in range(c_min, c_max):
                        if self.vision_radius_is_finite:
                            dr = r1 - r0
                            dc = c1 - c0
                            if (dr * dr + dc * dc) > self.vision_radius_sq:
                                continue

                        for r, c in self._bresenham(r0, c0, r1, c1):
                            if self.grid[r][c] == 1:
                                break
                            if (r, c) == (r1, c1):
                                visible.add((r1, c1))
                                break
                self.los_table[cell] = frozenset(visible)
        
        # --- NEW SQUARE GRID LOGIC ---
        elif self.los_type == 'square360':
            for cell in self.empty_cells:
                visible = set()
                r0, c0 = cell

                if self.vision_radius_is_finite:
                    radius_int = int(self.vision_radius)
                    r_min = max(0, r0 - radius_int)
                    r_max = min(self.rows, r0 + radius_int + 1)
                    c_min = max(0, c0 - radius_int)
                    c_max = min(self.cols, c0 + radius_int + 1)
                else:
                    r_min, r_max, c_min, c_max = 0, self.rows, 0, self.cols
                
                for r1 in range(r_min, r_max):
                    for c1 in range(c_min, c_max):
                        for r, c in self._bresenham(r0, c0, r1, c1):
                            if self.grid[r][c] == 1:
                                break
                            if (r, c) == (r1, c1):
                                visible.add((r1, c1))
                                break
                            
                self.los_table[cell] = frozenset(visible)

        else:
            # Raycasting for LOS4 / LOS8 with radius limit
            for cell in self.empty_cells:
                visible = set([cell])
                r, c = cell
                for dr, dc in self.vision_dirs:
                    nr, nc = r + dr, c + dc
                    current_dist = 1
                    # Keep moving along the ray as long as we are in bounds, 
                    # not hitting a wall, and under the radius limit
                    while (self.in_bounds(nr, nc) and 
                           self.grid[nr][nc] == 0 and 
                           current_dist <= self.vision_radius):
                        visible.add((nr, nc))
                        nr += dr
                        nc += dc
                        current_dist += 1
                self.los_table[cell] = frozenset(visible)

    def _precompute_apsp(self):
        """Precomputes All-Pairs Shortest Path (APSP) using parallel torch BFS.

        Stores result in self.apsp_matrix: a numpy [n, n] float32 array where
        apsp_matrix[i, j] is the shortest-path distance from cell i to cell j
        (indexed by self.cell_to_idx). Unreachable pairs have value n+1.
        """
        n = len(self.empty_cells_list)
        if n == 0:
            self.apsp_matrix = np.empty((0, 0), dtype=np.float32)
            return

        device = self._apsp_device
        INF = float(n + 1)

        # Build adjacency matrix
        adj_np = np.zeros((n, n), dtype=np.float32)
        for i, cell in enumerate(self.empty_cells_list):
            for nb in self.get_neighbors(cell):
                j = self.cell_to_idx.get(nb)
                if j is not None:
                    adj_np[i, j] = 1.0
        adj = torch.tensor(adj_np, dtype=torch.float32, device=device)

        # Parallel BFS from all n sources simultaneously.
        # frontier[s, v] = 1 means v is on the active BFS frontier from source s.
        dist = torch.full((n, n), INF, dtype=torch.float32, device=device)
        dist.fill_diagonal_(0.0)
        frontier = torch.eye(n, dtype=torch.float32, device=device)
        visited  = torch.eye(n, dtype=torch.bool,    device=device)

        step = 1
        while frontier.any():
            nf = (torch.mm(frontier, adj) > 0) & ~visited
            dist[nf] = float(step)
            visited  |= nf
            frontier  = nf.float()
            step += 1

        if device.type != 'cpu':
            torch.cuda.synchronize() if device.type == 'cuda' else torch.mps.synchronize()

        self.apsp_matrix = dist.cpu().numpy()  # [n, n] float32

    def _get_maximal_los_disjoint_pivots(self, unseen_mask):
        """Finds a maximal set of pivots that cannot see each other."""
        sorted_unseen = sorted(
            list(self._iter_mask_cells(unseen_mask)),
            key=lambda c: self.los_size_table[c],
            reverse=True,
        )
        pivots = []
        pivot_masks = []
        for c in sorted_unseen:
            c_mask = self.los_mask_table[c]
            is_disjoint = True
            for p_mask in pivot_masks:
                if c_mask & p_mask:
                    is_disjoint = False
                    break
            if is_disjoint:
                pivots.append(c)
                pivot_masks.append(c_mask)
        return pivots

    def heuristic_tsp(self, current_loc, unseen_mask):
        """
        Calculates the exact TSP Hamiltonian Path cost to visit all disjoint components.
        """
        # Full heuristic cache — same state always yields the same cost
        cache_key = (current_loc, unseen_mask)
        cached = self._heuristic_cache.get(cache_key)
        if cached is not None:
            return cached

        # Pivot cache — depends only on unseen_mask, not current_loc
        pivots = self._pivots_cache.get(unseen_mask)
        if pivots is None:
            pivots = self._get_maximal_los_disjoint_pivots(unseen_mask)
            self._pivots_cache[unseen_mask] = pivots

        if not pivots:
            self._heuristic_cache[cache_key] = 0
            return 0

        k = len(pivots)
        n = k + 1  # components: [current_loc] + pivots

        # Pivot-to-pivot distance submatrix — cached per unseen_mask (independent of current_loc)
        pivot_dm = self._pivot_dist_cache.get(unseen_mask)
        if pivot_dm is None:
            pivot_dm = np.full((k, k), np.inf, dtype=np.float32)
            np.fill_diagonal(pivot_dm, 0.0)
            for i in range(k):
                wi = self.los_idx_table[pivots[i]]
                if wi.size == 0:
                    continue
                for j in range(i + 1, k):
                    wj = self.los_idx_table[pivots[j]]
                    if wj.size == 0:
                        continue
                    d = float(self.apsp_matrix[wi[:, None], wj].min())
                    if d <= self.n_cells:
                        pivot_dm[i, j] = pivot_dm[j, i] = d
            self._pivot_dist_cache[unseen_mask] = pivot_dm

        # Distances from current_loc to each pivot.
        # Extract the full APSP row once (1-D slice) and then index into it — much faster
        # than the 2-D fancy-indexing used for general pivot pairs.
        loc_idx  = self.cell_to_idx[current_loc]
        loc_row  = self.apsp_matrix[loc_idx]           # shape (n_cells,)

        # loc_to_pivot as a plain Python list — faster than np.full for small k
        loc_to_pivot = [float('inf')] * k

        if n == 2:
            # Fast path: single pivot, no DP needed
            w = self.los_idx_table[pivots[0]]
            if w.size > 0:
                d = float(loc_row[w].min())
                result = d if d <= self.n_cells else 0
            else:
                result = 0
            self._heuristic_cache[cache_key] = result
            return result

        # Held-Karp TSP DP — use pivot_dm and loc_to_pivot directly;
        # no full dist_matrix array needed (eliminates np.full + fill_diagonal per call).
        for i, pivot in enumerate(pivots):
            w = self.los_idx_table[pivot]
            if w.size > 0:
                d = float(loc_row[w].min())
                if d <= self.n_cells:
                    loc_to_pivot[i] = d

        memo = {}
        for i in range(1, n):
            memo[(1 << i, i)] = loc_to_pivot[i - 1]

        for subset_size in range(2, n):
            for subset in itertools.combinations(range(1, n), subset_size):
                mask = 0
                for i in subset:
                    mask |= (1 << i)
                for next_node in subset:
                    prev_mask = mask ^ (1 << next_node)
                    min_cost = float('inf')
                    for prev_node in subset:
                        if prev_node != next_node:
                            # row/col offsets: component index i maps to pivot_dm index i-1
                            cost = memo.get((prev_mask, prev_node), float('inf')) + float(pivot_dm[prev_node - 1, next_node - 1])
                            if cost < min_cost:
                                min_cost = cost
                    memo[(mask, next_node)] = min_cost

        full_mask = (1 << n) - 2  # sum(1 << i for i in range(1, n))
        min_path_cost = min(memo.get((full_mask, i), float('inf')) for i in range(1, n))

        result = min_path_cost if min_path_cost != float('inf') else 0
        self._heuristic_cache[cache_key] = result
        return result
    

def solve_wrp_tsp_jf(wrp_grid):
    """A* search with Basic Expansion and TSP Heuristic."""
    start_loc = wrp_grid.start
    initial_seen_mask = wrp_grid.los_mask_table[start_loc]
    initial_unseen_mask = wrp_grid.all_seen_mask & ~initial_seen_mask

    pq = []
    tie_breaker = 0
    visited = {}

    # Parent-pointer path reconstruction: avoids copying an O(L) list per expansion.
    # node_locs[i] = location at node i; node_parent[i] = parent node index (-1 = root).
    node_locs   = [start_loc]
    node_parent = [-1]

    start_h = wrp_grid.heuristic_tsp(start_loc, initial_unseen_mask)
    heapq.heappush(pq, (start_h, tie_breaker, 0, start_loc, initial_seen_mask, 0))
    visited[(start_loc, initial_seen_mask)] = 0

    while pq:
        f, _, g, current_loc, seen_mask, curr_nid = heapq.heappop(pq)

        # Goal check — reconstruct path via parent pointers
        if seen_mask == wrp_grid.all_seen_mask:
            path = []
            nid = curr_nid
            while nid != -1:
                path.append(node_locs[nid])
                nid = node_parent[nid]
            path.reverse()
            return path, g

        # Prune stale heap entries
        if visited.get((current_loc, seen_mask), float('inf')) < g:
            continue

        # Basic Node Expansion
        for neighbor in wrp_grid.get_neighbors(current_loc):
            new_g = g + 1
            new_seen_mask = seen_mask | wrp_grid.los_mask_table[neighbor]
            state_key = (neighbor, new_seen_mask)

            if new_g < visited.get(state_key, float('inf')):
                visited[state_key] = new_g
                new_unseen_mask = wrp_grid.all_seen_mask & ~new_seen_mask
                h = wrp_grid.heuristic_tsp(neighbor, new_unseen_mask)

                new_nid = len(node_locs)
                node_locs.append(neighbor)
                node_parent.append(curr_nid)

                tie_breaker += 1
                heapq.heappush(pq, (new_g + h, tie_breaker, new_g, neighbor, new_seen_mask, new_nid))

    return None, float('inf')
