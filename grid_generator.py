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
        target_blocks = int(room_area * clutter_density / 4)
        if target_blocks <= 0:
            return

        inner_top = top + 1
        inner_bottom = top + height - 2
        inner_left = left + 1
        inner_right = left + width - 2
        if inner_bottom < inner_top or inner_right < inner_left:
            return

        min_center_dist = max(2, min(height, width) // 4)
        step_r = max(2, (height - 2) // 3)
        step_c = max(2, (width - 2) // 3)

        candidate_centers = []
        for base_r in range(inner_top, inner_bottom + 1, step_r):
            for base_c in range(inner_left, inner_right + 1, step_c):
                jitter_r = random.randint(-step_r // 3, step_r // 3)
                jitter_c = random.randint(-step_c // 3, step_c // 3)
                center_r = int(np.clip(base_r + jitter_r, inner_top, inner_bottom))
                center_c = int(np.clip(base_c + jitter_c, inner_left, inner_right))
                candidate_centers.append((center_r, center_c))
        random.shuffle(candidate_centers)

        attempts = 0
        placed = 0
        placed_centers = []
        max_attempts = max(target_blocks * 12, len(candidate_centers) * 3)
        while attempts < max_attempts and placed < target_blocks:
            attempts += 1
            block_h = random.choice([1, 2, 3, 4])
            block_w = random.choice([1, 2, 3, 4])

            if height <= block_h + 1 or width <= block_w + 1:
                continue

            if candidate_centers:
                center_r, center_c = candidate_centers.pop()
            else:
                center_r = random.randint(inner_top, inner_bottom)
                center_c = random.randint(inner_left, inner_right)

            min_by = inner_top
            max_by = top + height - block_h - 1
            min_bx = inner_left
            max_bx = left + width - block_w - 1
            if max_by < min_by or max_bx < min_bx:
                continue

            by = int(np.clip(center_r - block_h // 2, min_by, max_by))
            bx = int(np.clip(center_c - block_w // 2, min_bx, max_bx))

            block_center_r = by + block_h // 2
            block_center_c = bx + block_w // 2
            if any(
                (block_center_r - prev_r) ** 2 + (block_center_c - prev_c) ** 2 < min_center_dist ** 2
                for prev_r, prev_c in placed_centers
            ):
                continue

            if np.any(grid[by:by + block_h, bx:bx + block_w] == 1):
                continue

            grid[by:by + block_h, bx:bx + block_w] = 1
            placed += 1
            placed_centers.append((block_center_r, block_center_c))

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

    def generate_valid_grid(self, density=5, clutter_proba=0.75, free_ratio_threshold=0.5):
        """Generates realistic indoor-like maps with rooms, corridors, and clutter blocks.

        Args:
            density: controls clutter intensity (higher -> more obstacle blocks inside rooms).
            clutter_proba: whether to add clutter blocks inside rooms.
            free_ratio_threshold: minimum ratio of free cells to total cells to ensure enough navigable space.
        """
        min_side = min(self.rows, self.cols)
        if min_side < 8:
            raise ValueError("rows and cols should be at least 8 for room-corridor generation")

        # density = float(np.clip(density, 0.05, 0.9))
        max_tries = 250

        for _ in range(max_tries):
            grid = np.ones((self.rows, self.cols), dtype=np.int8)

            rooms = []
            room_target = random.randint(4, 32)

            for _ in range(room_target * 8):
                if len(rooms) >= room_target:
                    break

                room_h = random.randint(4, 16)
                room_w = random.randint(4, 16)

                if room_h >= self.rows - 1 or room_w >= self.cols - 1:
                    continue

                top = random.randint(0, self.rows - room_h)
                left = random.randint(0, self.cols - room_w)
                candidate = (top, left, room_h, room_w)

                if any(self._rooms_overlap(candidate, existing, padding=0) for existing in rooms):
                    continue

                self._carve_room(grid, top, left, room_h, room_w)
                rooms.append(candidate)

            if len(rooms) < 3:
                continue

            centers = [(r + h // 2, c + w // 2) for r, c, h, w in rooms]

            corridor_width = random.choice([1, 2, 3, 4, 5, 6, 7, 8])
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

            if clutter_proba > 0:
                clutter_density = density
                for room in rooms:
                    if random.random() < clutter_proba:  # chance to add clutter to a room
                        self._add_room_clutter(grid, room, clutter_density=clutter_density)

            if not self._is_fully_connected(grid):
                continue

            free_cells = np.argwhere(grid == 0)
            free_ratio = len(free_cells) / (self.rows * self.cols)
            if free_ratio < free_ratio_threshold:
                continue

            start_r, start_c = free_cells[random.randint(0, len(free_cells) - 1)]
            return grid, (int(start_r), int(start_c))

        raise RuntimeError("Failed to generate a connected realistic grid. Try increasing map size.")

    
    def _carve_zigzag_corridor(self, grid, r1, c1, r2, c2, width=1):
        """Carve a staircase/zigzag corridor between two points for irregular shapes."""
        num_segments = random.randint(3, 8)
        points = [(r1, c1)]
        for i in range(1, num_segments):
            t = i / num_segments
            mid_r = int(r1 + t * (r2 - r1) + random.randint(-3, 3))
            mid_c = int(c1 + t * (c2 - c1) + random.randint(-3, 3))
            mid_r = int(np.clip(mid_r, 0, self.rows - 1))
            mid_c = int(np.clip(mid_c, 0, self.cols - 1))
            points.append((mid_r, mid_c))
        points.append((r2, c2))

        for i in range(len(points) - 1):
            pr, pc = points[i]
            nr, nc = points[i + 1]
            if random.random() < 0.5:
                self._carve_h_corridor(grid, pr, pc, nc, width)
                self._carve_v_corridor(grid, nc, pr, nr, width)
            else:
                self._carve_v_corridor(grid, pc, pr, nr, width)
                self._carve_h_corridor(grid, nr, pc, nc, width)

    def _carve_random_walk_appendage(self, grid, start_r, start_c, length, width=1):
        """Carve a random-walk tendril from a starting free cell into wall territory."""
        r, c = start_r, start_c
        half = width // 2
        for _ in range(length):
            dr, dc = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            nr, nc = r + dr, c + dc
            if 1 <= nr < self.rows - 1 and 1 <= nc < self.cols - 1:
                r0 = max(0, nr - half)
                r1 = min(self.rows, nr + half + 1)
                c0 = max(0, nc - half)
                c1 = min(self.cols, nc + half + 1)
                grid[r0:r1, c0:c1] = 0
                r, c = nr, nc

    def _carve_l_shaped_room(self, grid, top, left, h, w):
        """Carve an L-shaped room by combining two overlapping rectangles."""
        # Vertical arm
        arm_w = max(2, w // 2 + random.randint(-1, 1))
        grid[top:top + h, left:left + arm_w] = 0
        # Horizontal arm at top or bottom
        arm_h = max(2, h // 2 + random.randint(-1, 1))
        if random.random() < 0.5:
            grid[top:top + arm_h, left:left + w] = 0
        else:
            grid[top + h - arm_h:top + h, left:left + w] = 0

    def _carve_t_shaped_room(self, grid, top, left, h, w):
        """Carve a T-shaped room."""
        stem_w = max(2, w // 3 + random.randint(-1, 1))
        stem_offset = (w - stem_w) // 2
        stem_h = max(2, h * 2 // 3)
        # Crossbar
        crossbar_h = max(2, h - stem_h)
        grid[top:top + crossbar_h, left:left + w] = 0
        # Stem
        grid[top + crossbar_h:top + crossbar_h + stem_h,
             left + stem_offset:left + stem_offset + stem_w] = 0

    def _add_wall_peninsulas(self, grid, num_peninsulas):
        """Grow wall peninsulas inward from boundary walls to create notches/crevices."""
        for _ in range(num_peninsulas):
            # Pick a random boundary cell that is wall
            side = random.choice(['top', 'bottom', 'left', 'right'])
            if side == 'top':
                r, c = 0, random.randint(0, self.cols - 1)
                dr, dc = 1, 0
            elif side == 'bottom':
                r, c = self.rows - 1, random.randint(0, self.cols - 1)
                dr, dc = -1, 0
            elif side == 'left':
                r, c = random.randint(0, self.rows - 1), 0
                dr, dc = 0, 1
            else:
                r, c = random.randint(0, self.rows - 1), self.cols - 1
                dr, dc = 0, -1

            pen_len = random.randint(2, max(3, min(self.rows, self.cols) // 3))
            pen_width = random.choice([1, 2])
            half = pen_width // 2

            for step in range(pen_len):
                nr, nc = r + dr * step, c + dc * step
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    for dw in range(-half, half + 1):
                        if dr == 0:  # moving horizontally, widen vertically
                            wr, wc = nr + dw, nc
                        else:  # moving vertically, widen horizontally
                            wr, wc = nr, nc + dw
                        if 0 <= wr < self.rows and 0 <= wc < self.cols:
                            grid[wr, wc] = 1

    def _erode_boundary(self, grid, iterations=1):
        """Erode free-space boundary cells randomly to create jagged/organic edges."""
        for _ in range(iterations):
            candidates = []
            for r in range(self.rows):
                for c in range(self.cols):
                    if grid[r, c] == 0:
                        # Check if adjacent to a wall
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = r + dr, c + dc
                            if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols or grid[nr, nc] == 1:
                                candidates.append((r, c))
                                break
            random.shuffle(candidates)
            num_erode = len(candidates) // random.randint(3, 6)
            for r, c in candidates[:num_erode]:
                grid[r, c] = 1

    def _ensure_simple_connectivity(self, grid):
        """THE TOPOLOGICAL TRICK: flood-fill walls from boundary via 8-connectivity.
        Any interior wall island not reachable = hole. Convert those to free space."""
        visited_walls = np.zeros((self.rows, self.cols), dtype=bool)
        queue = deque()
        # Seed from ALL boundary wall cells (not just corner)
        for r in range(self.rows):
            for c in range(self.cols):
                if (r == 0 or r == self.rows - 1 or c == 0 or c == self.cols - 1) and grid[r, c] == 1:
                    if not visited_walls[r, c]:
                        visited_walls[r, c] = True
                        queue.append((r, c))

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and grid[nr, nc] == 1 and not visited_walls[nr, nc]:
                    visited_walls[nr, nc] = True
                    queue.append((nr, nc))

        # Convert all unreached interior walls (holes) into free space
        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r, c] == 1 and not visited_walls[r, c]:
                    grid[r, c] = 0

    def generate_simple_polygon_grid(self, use_zig_zag_corridors=False):
        """
        Generates a strictly simply-connected polygon grid (no internal holes/pillars)
        with high geometric complexity: irregular rooms, zigzag corridors, random-walk
        tendrils, wall peninsulas, and boundary erosion for jagged organic edges.
        """
        max_tries = 200
        for _ in range(max_tries):
            grid = np.ones((self.rows, self.cols), dtype=np.int8)
            rooms = []
            room_target = random.randint(6, 48)

            # --- Phase 1: Carve diverse room shapes ---
            for _ in range(room_target * 10):
                if len(rooms) >= room_target:
                    break
                room_h = random.randint(3, max(4, self.rows // 2))
                room_w = random.randint(3, max(4, self.cols // 2))
                if room_h >= self.rows - 1 or room_w >= self.cols - 1:
                    continue
                top = random.randint(0, self.rows - room_h)
                left = random.randint(0, self.cols - room_w)
                candidate = (top, left, room_h, room_w)

                # Allow overlapping rooms (padding=0 or -1) for merged irregular regions
                padding = random.choice([-1, 0, 0, 1, 2])
                if any(self._rooms_overlap(candidate, existing, padding=padding) for existing in rooms):
                    continue

                # Randomly pick room shape
                shape_roll = random.random()
                if shape_roll < 0.4:
                    self._carve_room(grid, top, left, room_h, room_w)
                elif shape_roll < 0.65:
                    self._carve_l_shaped_room(grid, top, left, room_h, room_w)
                elif shape_roll < 0.85:
                    self._carve_t_shaped_room(grid, top, left, room_h, room_w)
                else:
                    # Plus-shaped room: two overlapping rectangles centered
                    arm1_h = room_h
                    arm1_w = max(2, room_w // 2)
                    arm2_h = max(2, room_h // 2)
                    arm2_w = room_w
                    cx, cy = top + room_h // 2, left + room_w // 2
                    r0 = max(0, cx - arm1_h // 2)
                    c0 = max(0, cy - arm1_w // 2)
                    grid[r0:min(self.rows, r0 + arm1_h), c0:min(self.cols, c0 + arm1_w)] = 0
                    r0 = max(0, cx - arm2_h // 2)
                    c0 = max(0, cy - arm2_w // 2)
                    grid[r0:min(self.rows, r0 + arm2_h), c0:min(self.cols, c0 + arm2_w)] = 0

                rooms.append(candidate)

            if len(rooms) < 3:
                continue

            # --- Phase 2: Connect rooms with varied corridor types ---
            centers = [(r + h // 2, c + w // 2) for r, c, h, w in rooms]
            connected = [0]
            unconnected = list(range(1, len(centers)))
            while unconnected:
                a = random.choice(connected)
                b = min(unconnected, key=lambda idx:
                        abs(centers[idx][0] - centers[a][0]) + abs(centers[idx][1] - centers[a][1]))
                r1, c1 = centers[a]
                r2, c2 = centers[b]
                corridor_width = random.choice([1, 1, 2, 2, 3])

                corridor_type = random.random()

                if use_zig_zag_corridors and corridor_type > 0.4:
                    self._carve_zigzag_corridor(grid, r1, c1, r2, c2, corridor_width)
                else:
                    # Standard L-shaped
                    if random.random() < 0.5:
                        self._carve_h_corridor(grid, r1, c1, c2, corridor_width)
                        self._carve_v_corridor(grid, c2, r1, r2, corridor_width)
                    else:
                        self._carve_v_corridor(grid, c1, r1, r2, corridor_width)
                        self._carve_h_corridor(grid, r2, c1, c2, corridor_width)

                connected.append(b)
                unconnected.remove(b)

            # Extra cross-links for more intricate shapes
            extra_links = random.randint(2, max(3, len(rooms) // 2))
            for _ in range(extra_links):
                i, j = random.sample(range(len(centers)), 2)
                r1, c1 = centers[i]
                r2, c2 = centers[j]
                w = random.choice([1, 1, 2])
                if use_zig_zag_corridors and random.random() < 0.5:
                    self._carve_zigzag_corridor(grid, r1, c1, r2, c2, w)
                else:
                    if random.random() < 0.5:
                        self._carve_h_corridor(grid, r1, c1, c2, w)
                        self._carve_v_corridor(grid, c2, r1, r2, w)
                    else:
                        self._carve_v_corridor(grid, c1, r1, r2, w)
                        self._carve_h_corridor(grid, r2, c1, c2, w)

            # --- Phase 3: Add random-walk tendrils from existing free-space boundary ---
            free_cells = list(zip(*np.where(grid == 0)))
            if free_cells:
                num_tendrils = random.randint(3, max(4, len(rooms)))
                for _ in range(num_tendrils):
                    sr, sc = random.choice(free_cells)
                    tendril_len = random.randint(4, max(5, min(self.rows, self.cols) // 2))
                    tendril_w = random.choice([1, 1, 2])
                    self._carve_random_walk_appendage(grid, sr, sc, tendril_len, tendril_w)

            # --- Phase 4: Add wall peninsulas (notches/crevices from boundary) ---
            num_peninsulas = random.randint(2, max(3, min(self.rows, self.cols) // 4))
            self._add_wall_peninsulas(grid, num_peninsulas)

            # --- Phase 5: Boundary erosion for jagged organic edges ---
            if random.random() < 0.6:
                self._erode_boundary(grid, iterations=random.randint(1, 3))

            # --- Phase 6: Ensure simple connectivity (remove internal holes) ---
            self._ensure_simple_connectivity(grid)

            if not self._is_fully_connected(grid):
                continue

            free_cells = np.argwhere(grid == 0)
            free_ratio = len(free_cells) / (self.rows * self.cols)
            if free_ratio < 0.20 or free_ratio > 0.85:
                continue

            start_r, start_c = free_cells[random.randint(0, len(free_cells) - 1)]
            return grid, (int(start_r), int(start_c))

        raise RuntimeError("Failed to generate complex simple polygon. Try increasing map size.")

    