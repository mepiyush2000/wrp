import heapq
import random
import itertools
from collections import deque

class WRPSolverTSPJF:
    def __init__(self, grid, start):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.start = start
        self.empty_cells = set()
        
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == 0:
                    self.empty_cells.add((r, c))
                    
        self.los_table = {}
        self.apsp_table = {}
        
        self._precompute_los4()
        self._precompute_apsp()

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def get_neighbors(self, loc):
        r, c = loc
        neighbors = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc) and self.grid[nr][nc] == 0:
                neighbors.append((nr, nc))
        return neighbors

    def _precompute_los4(self):
        """Precomputes the 4-way LOS for every empty cell."""
        for cell in self.empty_cells:
            visible = set([cell])
            r, c = cell
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                while self.in_bounds(nr, nc) and self.grid[nr][nc] == 0:
                    visible.add((nr, nc))
                    nr += dr
                    nc += dc
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
        # sorted_unseen = sorted(list(unseen), key=lambda c: len(self.los_table[c]))
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
        Uses Held-Karp Dynamic Programming for fast exact calculation.
        """
        pivots = self._get_maximal_los_disjoint_pivots(unseen)
        if not pivots:
            return 0
            
        # Abstract GDLS2: components are AgentCell (idx 0) + Pivots (idx 1 to n)
        components = [current_loc] + pivots
        n = len(components)
        
        # Build distance matrix
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

        # Base case for Held-Karp: Only AgentCell and 1 Pivot
        if n == 2:
            return dist_matrix[0][1] if dist_matrix[0][1] != float('inf') else 0

        # Held-Karp TSP formulation for minimum Hamiltonian path starting at 0
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
