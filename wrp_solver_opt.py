import heapq
import random
import itertools
from collections import deque

import heapq
import itertools
from collections import deque

class WRPSolverTSPJF:
    # Set default vision_radius to infinity for standard WRP
    def __init__(self, grid, start, los_type='los4', vision_radius=float('inf')):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.start = start
        self.los_type = los_type.lower()
        self.vision_radius = vision_radius # <-- Store the radius
        self.empty_cells = set()
        
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == 0:
                    self.empty_cells.add((r, c))
                    
        self.los_table = {}
        self.apsp_table = {}
        
        # Determine vision directions for raycasting (if not using Bresenham)
        if self.los_type == 'los8':
            self.vision_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), 
                                (1, 1), (1, -1), (-1, 1), (-1, -1)]
        elif self.los_type == 'los4':
            self.vision_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            
        self._precompute_los()
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

    def _precompute_los(self):
        """Precomputes the LOS for every empty cell with an optional radius limit."""
        if self.los_type == 'bresenham':
            for cell in self.empty_cells:
                visible = set()
                r0, c0 = cell
                
                # Bounding box for radius optimization
                r_min = max(0, int(r0 - self.vision_radius))
                r_max = min(self.rows, int(r0 + self.vision_radius) + 1)
                c_min = max(0, int(c0 - self.vision_radius))
                c_max = min(self.cols, int(c0 + self.vision_radius) + 1)
                
                # We MUST iterate over the bounding box, casting rays to WALLS too
                for r1 in range(r_min, r_max):
                    for c1 in range(c_min, c_max):
                        # Fast Euclidean check
                        target_dist = ((r1 - r0)**2 + (c1 - c0)**2)**0.5
                        if target_dist > self.vision_radius:
                            continue
                            
                        # Trace the Bresenham line and add ALL intermediate floor cells
                        for r, c in self._bresenham(r0, c0, r1, c1):
                            if not self.in_bounds(r, c):
                                break
                            if self.grid[r][c] == 1:
                                break  # Wall: target unreachable, drop the ray
                            if (r, c) == (r1, c1):
                                visible.add((r, c))  # Target reached cleanly
                                break
                        # Intermediate empty cell: do NOT mark
                self.los_table[cell] = frozenset(visible)
        
        # --- NEW SQUARE GRID LOGIC ---
        elif self.los_type == 'square360':
            for cell in self.empty_cells:
                visible = set()
                r0, c0 = cell
                
                r_min = max(0, int(r0 - self.vision_radius))
                r_max = min(self.rows, int(r0 + self.vision_radius) + 1)
                c_min = max(0, int(c0 - self.vision_radius))
                c_max = min(self.cols, int(c0 + self.vision_radius) + 1)
                
                for r1 in range(r_min, r_max):
                    for c1 in range(c_min, c_max):
                        # No Euclidean check needed
                        
                        for r, c in self._bresenham(r0, c0, r1, c1):
                            if not self.in_bounds(r, c):
                                break
                            if self.grid[r][c] == 1:
                                break
                            if (r, c) == (r1, c1):
                                visible.add((r, c))
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

    def _get_maximal_los_disjoint_pivots(self, unseen):
        """Finds a maximal set of pivots that cannot see each other."""
        sorted_unseen = sorted(list(unseen), key=lambda c: len(self.los_table[c]), reverse=True)
        pivots = []
        for c in sorted_unseen:
            is_disjoint = True
            for p in pivots:
                if not self.los_table[c].isdisjoint(self.los_table[p]):
                    is_disjoint = False
                    break
            if is_disjoint:
                pivots.append(c)
        return pivots

    def heuristic_tsp(self, current_loc, unseen):
        """
        Calculates the exact TSP Hamiltonian Path cost to visit all disjoint components.
        """
        pivots = self._get_maximal_los_disjoint_pivots(unseen)
        if not pivots:
            return 0
            
        components = [current_loc] + pivots
        n = len(components)
        
        dist_matrix = [[float('inf')] * n for _ in range(n)]
        
        def min_comp_dist(comp1, comp2):
            watchers1 = [current_loc] if comp1 == current_loc else self.los_table[comp1]
            watchers2 = [current_loc] if comp2 == current_loc else self.los_table[comp2]
                
            min_d = float('inf')
            for w1 in watchers1:
                for w2 in watchers2:
                    if w2 in self.apsp_table.get(w1, {}):
                        min_d = min(min_d, self.apsp_table[w1][w2])
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
    initial_seen = wrp_grid.los_table[start_loc]
    initial_unseen = frozenset(wrp_grid.empty_cells - initial_seen)
    
    pq = []
    tie_breaker = 0
    visited = {}
    
    start_h = wrp_grid.heuristic_tsp(start_loc, initial_unseen)
    heapq.heappush(pq, (start_h, tie_breaker, 0, start_loc, frozenset(initial_seen), [start_loc]))
    visited[(start_loc, frozenset(initial_seen))] = 0
    
    while pq:
        f, _, g, current_loc, seen, path = heapq.heappop(pq)
        
        # Goal check
        if len(seen) == len(wrp_grid.empty_cells):
            return path, g
            
        # Optional: Prune if a cheaper path to this state was found
        if visited.get((current_loc, seen), float('inf')) < g:
            continue
            
        # Basic Node Expansion
        for neighbor in wrp_grid.get_neighbors(current_loc):
            new_g = g + 1
            new_seen = seen.union(wrp_grid.los_table[neighbor])
            state_key = (neighbor, frozenset(new_seen))
            
            if new_g < visited.get(state_key, float('inf')):
                visited[state_key] = new_g
                new_unseen = wrp_grid.empty_cells - new_seen
                h = wrp_grid.heuristic_tsp(neighbor, new_unseen)
                
                tie_breaker += 1
                new_path = list(path) + [neighbor]
                heapq.heappush(pq, (new_g + h, tie_breaker, new_g, neighbor, new_seen, new_path))
                
    return None, float('inf')
