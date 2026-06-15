import heapq
import itertools
import math
from collections import deque

class WRPSolverTSPJF:
    # Set default vision_radius to infinity for standard WRP
    def __init__(self, grid, start, los_type='los4', vision_radius=float('inf')):
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
        self.apsp_table = {}

        self.empty_cells_list = sorted(self.empty_cells)
        self.cell_to_idx = {cell: idx for idx, cell in enumerate(self.empty_cells_list)}
        self.all_seen_mask = (1 << len(self.empty_cells_list)) - 1
        
        # Determine vision directions for raycasting (if not using Bresenham)
        if self.los_type == 'los8':
            self.vision_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), 
                                (1, 1), (1, -1), (-1, 1), (-1, -1)]
        elif self.los_type == 'los4':
            self.vision_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            
        self._precompute_los()
        self._build_los_masks()
        self._precompute_apsp()

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
        """Precomputes All-Pairs Shortest Path (APSP)."""
        for start_cell in self.empty_cells:
            self.apsp_table[start_cell] = {start_cell: 0}
            queue = deque([(start_cell, 0)])
            
            while queue:
                curr_cell, dist = queue.popleft()
                for neighbor in self.get_neighbors(curr_cell):
                    if neighbor not in self.apsp_table[start_cell]:
                        self.apsp_table[start_cell][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

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
        pivots = self._get_maximal_los_disjoint_pivots(unseen_mask)
        if not pivots:
            return 0
            
        components = [current_loc] + pivots
        n = len(components)
        
        dist_matrix = [[float('inf')] * n for _ in range(n)]
        
        def min_comp_dist(comp1, comp2):
            if comp1 == current_loc:
                watchers1_mask = 1 << self.cell_to_idx[current_loc]
            else:
                watchers1_mask = self.los_mask_table[comp1]

            if comp2 == current_loc:
                watchers2_mask = 1 << self.cell_to_idx[current_loc]
            else:
                watchers2_mask = self.los_mask_table[comp2]
                
            min_d = float('inf')
            for w1_idx in self._iter_mask_indices(watchers1_mask):
                w1 = self.empty_cells_list[w1_idx]
                dist_from_w1 = self.apsp_table.get(w1, {})
                for w2_idx in self._iter_mask_indices(watchers2_mask):
                    w2 = self.empty_cells_list[w2_idx]
                    if w2 in dist_from_w1:
                        d = dist_from_w1[w2]
                        if d < min_d:
                            min_d = d
            return min_d

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    dist_matrix[i][j] = 0
                else:
                    d = min_comp_dist(components[i], components[j])
                    dist_matrix[i][j] = d
                    dist_matrix[j][i] = d

        if n == 2:
            return dist_matrix[0][1] if dist_matrix[0][1] != float('inf') else 0

        memo = {}
        for i in range(1, n):
            memo[(1 << i, i)] = dist_matrix[0][i]

        for subset_size in range(2, n):
            for subset in itertools.combinations(range(1, n), subset_size):
                mask = sum(1 << i for i in subset)
                for next_node in subset:
                    prev_mask = mask ^ (1 << next_node)
                    
                    min_cost = float('inf')
                    for prev_node in subset:
                        if prev_node != next_node:
                            cost = memo.get((prev_mask, prev_node), float('inf')) + dist_matrix[prev_node][next_node]
                            if cost < min_cost:
                                min_cost = cost
                    memo[(mask, next_node)] = min_cost

        full_mask = sum(1 << i for i in range(1, n))
        min_path_cost = min(memo.get((full_mask, i), float('inf')) for i in range(1, n))
        
        return min_path_cost if min_path_cost != float('inf') else 0
    

def solve_wrp_tsp_jf(wrp_grid):
    """A* search with Basic Expansion and TSP Heuristic."""
    start_loc = wrp_grid.start
    initial_seen_mask = wrp_grid.los_mask_table[start_loc]
    initial_unseen_mask = wrp_grid.all_seen_mask & ~initial_seen_mask
    
    pq = []
    tie_breaker = 0
    visited = {}
    
    start_h = wrp_grid.heuristic_tsp(start_loc, initial_unseen_mask)
    heapq.heappush(pq, (start_h, tie_breaker, 0, start_loc, initial_seen_mask, [start_loc]))
    visited[(start_loc, initial_seen_mask)] = 0
    
    while pq:
        f, _, g, current_loc, seen_mask, path = heapq.heappop(pq)
        
        # Goal check
        if seen_mask == wrp_grid.all_seen_mask:
            return path, g
            
        # Optional: Prune if a cheaper path to this state was found
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
                
                tie_breaker += 1
                new_path = list(path) + [neighbor]
                heapq.heappush(pq, (new_g + h, tie_breaker, new_g, neighbor, new_seen_mask, new_path))
                
    return None, float('inf')
