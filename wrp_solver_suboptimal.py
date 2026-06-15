from wrp_solver_opt import WRPSolverTSPJF, solve_wrp_tsp_jf
 
import heapq
import itertools
import math
from collections import deque

# ===========================================================================
# 2. JUMP-TO-FRONTIER EXTENSION
# ===========================================================================
class WRPSolverJF(WRPSolverTSPJF):
    def __init__(self, grid, start, los_type='los4', vision_radius=float('inf')):
        super().__init__(grid, start, los_type, vision_radius)
        self._build_inverse_los()          # who can SEE each cell (true watchers)
        self._precompute_frontier_watchers()
        self._build_apsp_parents()         # to reconstruct corridors for jumps
        self._jump_los_cache = {}
 
    # --- precomputation ---------------------------------------------------
    def _build_inverse_los(self):
        """seen_by_mask[p] = bitmask of cells that can SEE p (transpose of LOS).
 
        Note: for los4/los8 this equals los_mask_table[p] (symmetric), but for
        bresenham/square360 LOS is asymmetric, so we build the real inverse."""
        n = len(self.empty_cells_list)
        seen_by = [0] * n
        for watcher, vis_mask in self.los_mask_table.items():
            w_bit = 1 << self.cell_to_idx[watcher]
            for idx in self._iter_mask_indices(vis_mask):
                seen_by[idx] |= w_bit
        self.seen_by_mask = {self.empty_cells_list[i]: seen_by[i] for i in range(n)}
 
    def _precompute_frontier_watchers(self):
        """frontier_watchers[p] = watchers of p that border a traversable
        non-watcher cell (i.e. entry points into p's visibility region).
        State-independent -- depends only on LOS geometry."""
        self.frontier_watchers = {}
        for p in self.empty_cells:
            wmask = self.seen_by_mask[p]
            fw = []
            for w in self._iter_mask_cells(wmask):
                wr, wc = w
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = wr + dr, wc + dc
                    if not self.in_bounds(nr, nc) or self.grid[nr][nc] == 1:
                        continue  # walls/edges are not entry points
                    if not (wmask & (1 << self.cell_to_idx[(nr, nc)])):
                        fw.append(w)
                        break
            # Defensive fallback (rare): a fully enclosed region with no border.
            self.frontier_watchers[p] = fw if fw else list(self._iter_mask_cells(wmask))
 
    def _build_apsp_parents(self):
        """BFS parent pointers so a shortest path for any jump is reconstructable."""
        self.apsp_parent = {}
        for start_cell in self.empty_cells:
            parent = {start_cell: None}
            queue = deque([start_cell])
            while queue:
                curr = queue.popleft()
                for nb in self.get_neighbors(curr):
                    if nb not in parent:
                        parent[nb] = curr
                        queue.append(nb)
            self.apsp_parent[start_cell] = parent
 
    # --- jump machinery ---------------------------------------------------
    def _shortest_path_cells(self, src, dst):
        parent = self.apsp_parent[src]
        path = []
        node = dst
        while node is not None:
            path.append(node)
            node = parent.get(node)
        path.reverse()
        return path  # [src, ..., dst]
 
    def _jump_los_mask(self, loc, target):
        """OR of LOS over the canonical shortest path loc->target (cached).
        Depends only on (loc, target), independent of current seen state."""
        key = (loc, target)
        cached = self._jump_los_cache.get(key)
        if cached is not None:
            return cached
        mask = 0
        for cell in self._shortest_path_cells(loc, target):
            mask |= self.los_mask_table[cell]
        self._jump_los_cache[key] = mask
        return mask
 
    def jf_expand(self, loc, seen_mask, df=float('inf')):
        """Yield (jump_point, edge_cost, new_seen_mask) for each frontier-watcher
        jump from `loc`. With df < inf keep only jumps within df * nearest cost."""
        unseen_mask = self.all_seen_mask & ~seen_mask
        pivots = self._get_maximal_los_disjoint_pivots(unseen_mask)
        dist_from_loc = self.apsp_table.get(loc, {})
 
        candidates, seen_targets = [], set()
        for p in pivots:
            for w in self.frontier_watchers[p]:
                if w in seen_targets:
                    continue
                seen_targets.add(w)
                d = dist_from_loc.get(w)
                if d is None or d == 0:        # unreachable or self
                    continue
                candidates.append((d, w))
 
        if not candidates:
            return
        if df != float('inf'):
            cutoff = df * min(d for d, _ in candidates)
            candidates = [(d, w) for d, w in candidates if d <= cutoff]
 
        for d, w in candidates:
            yield (w, d, seen_mask | self._jump_los_mask(loc, w))
 
    # --- heuristics (shared distance matrix; MST is the fast alternative) --
    def _min_comp_dist(self, current_loc, comp1, comp2):
        """Min grid distance between two components. A component's 'watchers'
        are the cells that can SEE its pivot (true inverse LOS via seen_by_mask,
        correct even for asymmetric bresenham/square360 LOS). The current-loc
        component is just {current_loc}."""
        w1m = (1 << self.cell_to_idx[current_loc]) if comp1 == current_loc else self.seen_by_mask[comp1]
        w2m = (1 << self.cell_to_idx[current_loc]) if comp2 == current_loc else self.seen_by_mask[comp2]
        min_d = float('inf')
        for i1 in self._iter_mask_indices(w1m):
            dfrom = self.apsp_table.get(self.empty_cells_list[i1], {})
            if not dfrom:
                continue
            for i2 in self._iter_mask_indices(w2m):
                d = dfrom.get(self.empty_cells_list[i2])
                if d is not None and d < min_d:
                    min_d = d
        return min_d
 
    def _build_component_dist_matrix(self, current_loc, pivots):
        components = [current_loc] + pivots
        n = len(components)
        dist = [[0 if i == j else float('inf') for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = self._min_comp_dist(current_loc, components[i], components[j])
                dist[i][j] = dist[j][i] = d
        return components, dist
 
    def heuristic_tsp(self, current_loc, unseen_mask):
        """Exact min Hamiltonian-path lower bound (Held-Karp). Tightest, but
        O(n^2 * 2^n) in the number of pivots -- the df>1 bottleneck.
        Overrides the base version to use true inverse-LOS watchers."""
        pivots = self._get_maximal_los_disjoint_pivots(unseen_mask)
        if not pivots:
            return 0
        components, dist = self._build_component_dist_matrix(current_loc, pivots)
        n = len(components)
        if n == 2:
            return dist[0][1] if dist[0][1] != float('inf') else 0
        memo = {}
        for i in range(1, n):
            memo[(1 << i, i)] = dist[0][i]
        for size in range(2, n):
            for subset in itertools.combinations(range(1, n), size):
                mask = sum(1 << i for i in subset)
                for nxt in subset:
                    prev_mask = mask ^ (1 << nxt)
                    best = float('inf')
                    for prev in subset:
                        if prev != nxt:
                            cost = memo.get((prev_mask, prev), float('inf')) + dist[prev][nxt]
                            if cost < best:
                                best = cost
                    memo[(mask, nxt)] = best
        full = sum(1 << i for i in range(1, n))
        best = min(memo.get((full, i), float('inf')) for i in range(1, n))
        return best if best != float('inf') else 0
 
    def heuristic_mst(self, current_loc, unseen_mask):
        """MST lower bound over {current_loc} U pivots. A Hamiltonian path is a
        spanning tree, so MST <= TSP-path <= true remaining cost -> admissible,
        and only O(n^2) (Prim). Weaker bound than TSP (=> more node expansions),
        but no exponential per-node cost -- this is what unblocks df>1 / large
        grids for bulk dataset generation."""
        pivots = self._get_maximal_los_disjoint_pivots(unseen_mask)
        if not pivots:
            return 0
        components, dist = self._build_component_dist_matrix(current_loc, pivots)
        n = len(components)
        INF = float('inf')
        in_tree = [False] * n
        best = [INF] * n
        best[0] = 0
        total = 0
        for _ in range(n):
            u, u_cost = -1, INF
            for v in range(n):
                if not in_tree[v] and best[v] < u_cost:
                    u, u_cost = v, best[v]
            if u == -1 or u_cost == INF:
                break  # remaining pivots unreachable; partial MST is still a LB
            in_tree[u] = True
            total += u_cost
            row = dist[u]
            for v in range(n):
                if not in_tree[v] and row[v] < best[v]:
                    best[v] = row[v]
        return total
 
 
def solve_wrp_jf(wrp_grid, weight=1.0, df=float('inf'), heuristic='tsp'):
    """A* / Weighted A* over Jump-to-Frontier expansion.
 
      weight=1.0, df=inf : closest to optimal (IW, so not formally guaranteed)
      weight>1.0         : bounded layer, solution <= weight * cost_of_solution
      df<inf  (df>=1)    : prune far jumps -> much smaller branching, lower quality
      heuristic='tsp'    : exact Held-Karp; tightest but exponential in pivots
      heuristic='mst'    : MST lower bound; polynomial -> use for df>1 / big grids
 
    Returns (route, cost) where `route` is the FULL step-by-step path with all
    corridors expanded, ready to rasterize into a flow field."""
    h_fn = wrp_grid.heuristic_mst if heuristic == 'mst' else wrp_grid.heuristic_tsp
    start_loc = wrp_grid.start
    initial_seen = wrp_grid.los_mask_table[start_loc]
    start_state = (start_loc, initial_seen)
 
    pq = []
    tie = 0
    g_best = {start_state: 0}
    came_from = {start_state: None}        # state -> (prev_state, jump_target)
 
    h0 = h_fn(start_loc, wrp_grid.all_seen_mask & ~initial_seen)
    heapq.heappush(pq, (weight * h0, tie, 0, start_loc, initial_seen))
 
    while pq:
        f, _, g, loc, seen = heapq.heappop(pq)
        if seen == wrp_grid.all_seen_mask:
            return _reconstruct_route(wrp_grid, came_from, (loc, seen)), g
        if g_best.get((loc, seen), float('inf')) < g:
            continue
        for w, cost, new_seen in wrp_grid.jf_expand(loc, seen, df=df):
            new_g = g + cost
            key = (w, new_seen)
            if new_g < g_best.get(key, float('inf')):
                g_best[key] = new_g
                came_from[key] = ((loc, seen), w)
                h = h_fn(w, wrp_grid.all_seen_mask & ~new_seen)
                tie += 1
                heapq.heappush(pq, (new_g + weight * h, tie, new_g, w, new_seen))
    return None, float('inf')
 
 
def _reconstruct_route(wrp_grid, came_from, goal_state):
    jumps = []  # (from_loc, to_loc) in reverse
    state = goal_state
    while came_from.get(state) is not None:
        prev_state, target = came_from[state]
        jumps.append((prev_state[0], target))
        state = prev_state
    jumps.reverse()
    route = [wrp_grid.start]
    for from_loc, to_loc in jumps:
        route.extend(wrp_grid._shortest_path_cells(from_loc, to_loc)[1:])
    return route
 
 
# ===========================================================================
# 3. VALIDATION / SMOKE TEST
# ===========================================================================
def _verify_route(grid_obj, route):
    seen = 0
    for cell in route:
        seen |= grid_obj.los_mask_table[cell]
    return seen == grid_obj.all_seen_mask
 
 
if __name__ == '__main__':
    import time, random
 
    def random_grid(n, density, seed):
        rnd = random.Random(seed)
        g = [[1 if rnd.random() < density else 0 for _ in range(n)] for _ in range(n)]
        g[0][0] = 0
        return g
 
    print("=== Correctness: basic A* == JF(TSP) == JF(MST), all at optimal ===")
    for seed in range(8):
        g = random_grid(7, 0.18, seed)
        base = WRPSolverTSPJF(g, (0, 0), 'los4')
        jf = WRPSolverJF(g, (0, 0), 'los4')
        _, cb = solve_wrp_tsp_jf(base)
        rt, ct = solve_wrp_jf(jf, 1.0, float('inf'), 'tsp')
        rm, cm = solve_wrp_jf(jf, 1.0, float('inf'), 'mst')
        if cm == float('inf'):
            print(f"  seed {seed}: unsolvable (all inf)  match={cb == ct == cm}")
            assert cb == ct == cm
            continue
        ok = _verify_route(jf, rm) and _verify_route(jf, rt)
        print(f"  seed {seed}: basic={cb} TSP={ct} MST={cm}  match={cb == ct == cm}  valid={ok}")
        assert cb == ct == cm and ok
 
    print("\n=== Heuristic speed: TSP vs MST at df=1.0 (same optimal cost) ===")
    g = random_grid(20, 0.25, 3)  # known-solvable instance
    jf = WRPSolverJF(g, (0, 0), 'los4')
    t = time.time(); rt, ct = solve_wrp_jf(jf, 1.0, 1.0, 'tsp'); tt = time.time() - t
    t = time.time(); rm, cm = solve_wrp_jf(jf, 1.0, 1.0, 'mst'); tm = time.time() - t
    print(f"  TSP df=1.0: cost={ct} time={tt:.3f}s")
    print(f"  MST df=1.0: cost={cm} time={tm:.3f}s  (~{tt / max(tm, 1e-6):.0f}x faster, same cost={ct == cm})")
 
    print("\n=== Knobs: MST unblocks df>1; weight trades quality for speed ===")
    presets = [("fast bulk", 5.0, 1.0), ("balanced", 4.0, 1.5), ("wider/fast", 8.0, 1.5)]
    for name, w, df in presets:
        t = time.time(); r, c = solve_wrp_jf(jf, w, df, 'mst'); dt = time.time() - t
        if r:
            print(f"  {name:12s} (weight={w}, df={df}): cost={c} time={dt:.3f}s valid={_verify_route(jf, r)}")
        else:
            print(f"  {name:12s}: unsolvable instance")
    # Note: lowering weight toward 1 at df>1 finds shorter routes but costs more
    # time (e.g. weight=2, df=1.5 on this grid reaches cost ~115 but is ~100x slower).
 