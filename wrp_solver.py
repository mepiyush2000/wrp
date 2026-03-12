import heapq
import random
from collections import deque

class WRPSolver:
    def __init__(self, grid, start):
        """
        grid: 2D list where 0 is empty (traversable) and 1 is an obstacle.
        start: tuple (r, c) representing the start state.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.start = start
        self.empty_cells = set()
        
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == 0:
                    self.empty_cells.add((r, c))
                    
        # Preprocessing tables
        self.los_table = {}
        self.apsp_table = {}
        
        self._precompute_los4()
        self._precompute_apsp()

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _precompute_los4(self):
        """Precomputes the 4-way LOS for every empty cell."""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for cell in self.empty_cells:
            visible = set([cell])
            r, c = cell
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                # Keep going in the same cardinal direction until an obstacle or boundary
                while self.in_bounds(nr, nc) and self.grid[nr][nc] == 0:
                    visible.add((nr, nc))
                    nr += dr
                    nc += dc
            self.los_table[cell] = visible

    def _precompute_apsp(self):
        """Precomputes All-Pairs Shortest Path using BFS (since edge weights are 1)."""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for start_cell in self.empty_cells:
            self.apsp_table[start_cell] = {start_cell: 0}
            queue = deque([(start_cell, 0)])
            visited = set([start_cell])
            
            while queue:
                (curr_r, curr_c), dist = queue.popleft()
                for dr, dc in directions:
                    nr, nc = curr_r + dr, curr_c + dc
                    neighbor = (nr, nc)
                    if neighbor in self.empty_cells and neighbor not in visited:
                        visited.add(neighbor)
                        self.apsp_table[start_cell][neighbor] = dist + 1
                        queue.append((neighbor, dist + 1))

    def heuristic_singleton(self, current_location, unseen_cells):
        """
        Calculates the maximum over all unseen pivots of the minimum distance 
        to a cell that has LOS to the pivot.
        """
        if not unseen_cells:
            return 0
            
        max_h = 0
        for pivot in unseen_cells:
            # Find minimum distance from current location to any cell q that sees pivot p
            min_dist = float('inf')
            for q in self.los_table[pivot]:
                # If q is reachable from current_location
                if q in self.apsp_table[current_location]: 
                    dist = self.apsp_table[current_location][q]
                    if dist < min_dist:
                        min_dist = dist
            
            if min_dist > max_h:
                max_h = min_dist
                
        return max_h

def solve_wrp_a_star(wrp_grid):
    """Solves the WRP using A* search with the Singleton heuristic."""
    start_loc = wrp_grid.start
    
    # Root node setup
    initial_seen = wrp_grid.los_table[start_loc]
    initial_unseen = wrp_grid.empty_cells - initial_seen
    
    # Priority queue stores tuples: (f_score, tie_breaker, g_score, location, seen_cells, path)
    tie_breaker = 0
    pq = []
    
    # Pruning table: stores the minimum g_score for a specific (location, frozenset(seen))
    visited = {}
    
    start_h = wrp_grid.heuristic_singleton(start_loc, initial_unseen)
    heapq.heappush(pq, (start_h, tie_breaker, 0, start_loc, frozenset(initial_seen), [start_loc]))
    
    visited[(start_loc, frozenset(initial_seen))] = 0
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while pq:
        f, _, g, current_loc, seen, path = heapq.heappop(pq)
        
        # Goal check: all empty cells are seen
        if len(seen) == len(wrp_grid.empty_cells):
            return path, g
            
        # Optional: Prune if we've found a better path to this exact state in the meantime
        if visited.get((current_loc, seen), float('inf')) < g:
            continue
            
        # Expand node
        r, c = current_loc
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)
            
            if neighbor in wrp_grid.empty_cells:
                new_g = g + 1
                new_seen = seen.union(wrp_grid.los_table[neighbor])
                
                state_key = (neighbor, new_seen)
                
                # Only add to open list if this is a newly discovered state or a strictly better path
                if new_g < visited.get(state_key, float('inf')):
                    visited[state_key] = new_g
                    new_unseen = wrp_grid.empty_cells - new_seen
                    h = wrp_grid.heuristic_singleton(neighbor, new_unseen)
                    f = new_g + h
                    
                    tie_breaker += 1
                    new_path = list(path) + [neighbor]
                    heapq.heappush(pq, (f, tie_breaker, new_g, neighbor, new_seen, new_path))
                    
    return None, float('inf') # No path found