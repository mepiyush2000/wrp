import numpy as np
import random
from collections import deque

class WRPDataGenerator:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

    def _carve_room(self, grid, top, left, height, width):
        grid[top:top + height, left:left + width] = 0

    def _carve_h_corridor(self, grid, row, c0, c1, corridor_width=1):
        start_col, end_col = sorted((c0, c1))
        half = corridor_width // 2
        r0 = max(0, row - half)
        r1 = min(self.rows, row + half + 1)
        grid[r0:r1, start_col:end_col + 1] = 0

    def _carve_v_corridor(self, grid, col, r0, r1, corridor_width=1):
        start_row, end_row = sorted((r0, r1))
        half = corridor_width // 2
        c0 = max(0, col - half)
        c1 = min(self.cols, col + half + 1)
        grid[start_row:end_row + 1, c0:c1] = 0

    def _rooms_overlap(self, room_a, room_b, padding=2):
        ta, la, ha, wa = room_a
        tb, lb, hb, wb = room_b

        a_top = ta - padding
        a_left = la - padding
        a_bottom = ta + ha + padding
        a_right = la + wa + padding

        b_top = tb
        b_left = lb
        b_bottom = tb + hb
        b_right = lb + wb

        return not (a_right <= b_left or b_right <= a_left or a_bottom <= b_top or b_bottom <= a_top)

    def _add_room_clutter(self, grid, room, clutter_density=0.1):
        top, left, height, width = room
        room_area = height * width
        target_blocks = max(1, int(room_area * clutter_density / 4))

        attempts = 0
        placed = 0
        while attempts < target_blocks * 8 and placed < target_blocks:
            attempts += 1
            block_h = random.choice([1, 1, 2])
            block_w = random.choice([1, 1, 2])

            if height <= block_h + 1 or width <= block_w + 1:
                continue

            by = random.randint(top + 1, top + height - block_h - 1)
            bx = random.randint(left + 1, left + width - block_w - 1)

            if np.any(grid[by:by + block_h, bx:bx + block_w] == 1):
                continue

            grid[by:by + block_h, bx:bx + block_w] = 1
            placed += 1

    def _is_fully_connected(self, grid):
        """Ensures all empty cells can reach each other via cardinal moves[cite: 73, 74]."""
        empty_cells = [(r, c) for r in range(self.rows) for c in range(self.cols) if grid[r, c] == 0]
        if not empty_cells: return False
        start = empty_cells[0]
        queue = deque([start])
        visited = {start}
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and \
                   grid[nr, nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return len(visited) == len(empty_cells)

    def generate_valid_grid(self, density=0.2):
        """Generates realistic indoor-like maps with rooms, corridors, and clutter blocks.

        Args:
            density: controls clutter intensity (higher -> more obstacle blocks inside rooms).
        """
        min_side = min(self.rows, self.cols)
        if min_side < 8:
            raise ValueError("rows and cols should be at least 8 for room-corridor generation")

        density = float(np.clip(density, 0.05, 0.45))
        max_tries = 250

        for _ in range(max_tries):
            grid = np.ones((self.rows, self.cols), dtype=np.int8)

            rooms = []
            room_target = random.randint(6, 12)

            for _ in range(room_target * 8):
                if len(rooms) >= room_target:
                    break

                room_h = random.randint(4, 8)
                room_w = random.randint(4, 8)

                if room_h >= self.rows - 1 or room_w >= self.cols - 1:
                    continue

                top = random.randint(0, self.rows - room_h)
                left = random.randint(0, self.cols - room_w)
                candidate = (top, left, room_h, room_w)

                if any(self._rooms_overlap(candidate, existing, padding=2) for existing in rooms):
                    continue

                self._carve_room(grid, top, left, room_h, room_w)
                rooms.append(candidate)

            if len(rooms) < 3:
                continue

            centers = [(r + h // 2, c + w // 2) for r, c, h, w in rooms]

            corridor_width = random.choice([1, 1, 2])
            connected = [0]
            unconnected = list(range(1, len(centers)))
            while unconnected:
                a = random.choice(connected)
                b = min(unconnected, key=lambda idx: abs(centers[idx][0] - centers[a][0]) + abs(centers[idx][1] - centers[a][1]))

                r1, c1 = centers[a]
                r2, c2 = centers[b]

                if random.random() < 0.5:
                    self._carve_h_corridor(grid, r1, c1, c2, corridor_width)
                    self._carve_v_corridor(grid, c2, r1, r2, corridor_width)
                else:
                    self._carve_v_corridor(grid, c1, r1, r2, corridor_width)
                    self._carve_h_corridor(grid, r2, c1, c2, corridor_width)

                connected.append(b)
                unconnected.remove(b)

            extra_links = random.randint(1, max(2, len(rooms) // 3))
            for _ in range(extra_links):
                i, j = random.sample(range(len(centers)), 2)
                r1, c1 = centers[i]
                r2, c2 = centers[j]
                if random.random() < 0.5:
                    self._carve_h_corridor(grid, r1, c1, c2, corridor_width)
                else:
                    self._carve_v_corridor(grid, c1, r1, r2, corridor_width)

            clutter_density = 0.05 + 0.35 * density
            for room in rooms:
                if random.random() < 0.8:
                    self._add_room_clutter(grid, room, clutter_density=clutter_density)

            if not self._is_fully_connected(grid):
                continue

            free_cells = np.argwhere(grid == 0)
            free_ratio = len(free_cells) / (self.rows * self.cols)
            if free_ratio < 0.30:
                continue

            start_r, start_c = free_cells[random.randint(0, len(free_cells) - 1)]
            return grid, (int(start_r), int(start_c))

        raise RuntimeError("Failed to generate a connected realistic grid. Try increasing map size.")

    
    def generate_simple_polygon_grid(self, density=0.2):
        """
        Generates a strictly simply-connected polygon grid (no internal holes/pillars).
        This guarantees the Watchman Route Problem is solvable in Polynomial Time via DP.
        """
        # 1. Generate a standard grid but FORBID clutter (no internal blocks)
        max_tries = 50
        for _ in range(max_tries):
            # We use the existing logic but bypass the clutter addition
            grid = np.ones((self.rows, self.cols), dtype=np.int8)
            rooms = []
            room_target = random.randint(4, 20)

            for _ in range(room_target * 8):
                if len(rooms) >= room_target: break
                room_h, room_w = random.randint(2, 12), random.randint(2, 12)
                if room_h >= self.rows - 1 or room_w >= self.cols - 1: continue
                top, left = random.randint(0, self.rows - room_h), random.randint(0, self.cols - room_w)
                candidate = (top, left, room_h, room_w)
                if any(self._rooms_overlap(candidate, existing, padding=2) for existing in rooms): continue
                self._carve_room(grid, top, left, room_h, room_w)
                rooms.append(candidate)

            if len(rooms) < 3: continue

            # Connect rooms with corridors
            centers = [(r + h // 2, c + w // 2) for r, c, h, w in rooms]
            connected = [0]
            unconnected = list(range(1, len(centers)))
            while unconnected:
                a = random.choice(connected)
                b = min(unconnected, key=lambda idx: abs(centers[idx][0] - centers[a][0]) + abs(centers[idx][1] - centers[a][1]))
                r1, c1, r2, c2 = centers[a][0], centers[a][1], centers[b][0], centers[b][1]
                
                if random.random() < 0.5:
                    self._carve_h_corridor(grid, r1, c1, c2, 1)
                    self._carve_v_corridor(grid, c2, r1, r2, 1)
                else:
                    self._carve_v_corridor(grid, c1, r1, r2, 1)
                    self._carve_h_corridor(grid, r2, c1, c2, 1)
                connected.append(b)
                unconnected.remove(b)

            # 2. THE TOPOLOGICAL TRICK: Remove all internal loops to make it "Simply-Connected"
            # We flood-fill the walls (1s) starting from the top-left (0,0) which is always a wall.
            # Any '1' that the flood-fill cannot reach is an internal pillar creating a hole.
            visited_walls = set()
            queue = deque([(0, 0)])
            visited_walls.add((0, 0))
            
            while queue:
                r, c = queue.popleft()
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols and grid[nr, nc] == 1 and (nr, nc) not in visited_walls:
                        visited_walls.add((nr, nc))
                        queue.append((nr, nc))
            
            # Convert all unreached 1s (internal holes) into 0s (free space)
            for r in range(self.rows):
                for c in range(self.cols):
                    if grid[r, c] == 1 and (r, c) not in visited_walls:
                        grid[r, c] = 0 # Erase the hole!

            if not self._is_fully_connected(grid): continue
            
            free_cells = np.argwhere(grid == 0)
            start_r, start_c = free_cells[random.randint(0, len(free_cells) - 1)]
            return grid, (int(start_r), int(start_c))

        raise RuntimeError("Failed to generate simple polygon.")

    