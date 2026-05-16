import torch
from tqdm import tqdm
from collections import deque
from heapq import heappop, heappush
from PIL import Image, ImageDraw
from grid_generator import *
from wrp_solver_opt import *
from utils import *


def generate_training_data(grid, path, apply_smoothening = False):
    """Generates training data: input=(grid, current_cell, unseen_map), label=next_cell."""
    grid_tensor = torch.from_numpy(grid).float()  # (H, W)
    X = []
    y = []

    for i in range(len(path) - 1):
        # Channel 0: grid map (obstacles=1, free=0)
        # Already have grid_tensor

        # Channel 1: current cell one-hot
        current_cell_map = torch.zeros_like(grid_tensor)
        current_cell_map[path[i][0], path[i][1]] = 1.0

        # Channel 2: unseen map (1 = not yet visible, 0 = already seen)
        visibility = get_LOS4_visibility_map(grid, path[:i + 1])
        # Assuming grid has 1 for obstacles and 0 for free space:
        # (1 - grid) creates a mask of ONLY free space.
        unseen_map_numpy = (1 - visibility) * (1 - grid)
        unseen_map = torch.from_numpy(unseen_map_numpy).float()

        # Features: (3, H, W)
        features = torch.stack([grid_tensor, current_cell_map, unseen_map], dim=0)
        X.append(features)

        # Label: next cell one-hot (1, H, W)
        # next_cell_map = torch.zeros_like(grid_tensor)
        if apply_smoothening:
            next_cell_map = apply_spatial_smoothing(grid_tensor, current_cell_map, smooth_val=0.2)
            next_cell_map[path[i + 1][0], path[i + 1][1]] = 1.0
            next_cell_map[path[i][0], path[i][1]] = 0.0  # Ensure current cell is not labeled as next  
        else:
            next_cell_map = torch.zeros_like(grid_tensor)
            next_cell_map[path[i + 1][0], path[i + 1][1]] = 1.0

        y.append(next_cell_map.unsqueeze(0))

    return torch.stack(X), torch.stack(y)

def _solve_grid(grid, start, los_type = "los4", vision_radius = float('inf')):
    solver = WRPSolverTSPJF(grid, start, los_type=los_type, vision_radius=vision_radius)
    return solve_wrp_tsp_jf(solver)

def generate_N_training_data(num_samples, grid_size=(16, 16), density=5, timeout=900):
    X_list = []
    y_list = []
    skipped = 0
    
    for _ in tqdm(range(num_samples)):
        # Generate a random grid and path
        gen = WRPDataGenerator(*grid_size)
        grid, start = gen.generate_valid_grid(density=density)
        # grid, start = gen.generate_simple_polygon_grid(density=density)
        
        try:
            path_opt, _ = run_with_timeout(_solve_grid, args=(grid, start), timeout=timeout)
        except TimeoutError:
            skipped += 1
            continue
        
        # Generate training data from the path
        X, y = generate_training_data(grid, path_opt)
        X_list.append(X)
        y_list.append(y)
    
    if skipped:
        print(f"Skipped {skipped}/{num_samples} samples due to timeout ({timeout}s)")
    
    return torch.cat(X_list), torch.cat(y_list)


def augment_data(X, y):
    """Augment training data with all 8 dihedral transforms (4 rotations × 2 flips).
    
    Both X (B, C, H, W) and y (B, 1, H, W) are transformed identically
    so the spatial correspondence is preserved.
    """
    augmented_X = [X]
    augmented_y = [y]
    
    # dims=[-2, -1] rotate in the H, W plane
    for k in range(1, 4):  # 90°, 180°, 270°
        augmented_X.append(torch.rot90(X, k, dims=[-2, -1]))
        augmented_y.append(torch.rot90(y, k, dims=[-2, -1]))
    
    # Horizontal flip
    X_flip = torch.flip(X, dims=[-1])
    y_flip = torch.flip(y, dims=[-1])
    augmented_X.append(X_flip)
    augmented_y.append(y_flip)
    
    # Horizontal flip + 3 rotations
    for k in range(1, 4):
        augmented_X.append(torch.rot90(X_flip, k, dims=[-2, -1]))
        augmented_y.append(torch.rot90(y_flip, k, dims=[-2, -1]))
    
    X_aug = torch.cat(augmented_X, dim=0)
    y_aug = torch.cat(augmented_y, dim=0)
    
    # Shuffle so augmented versions aren't grouped together
    perm = torch.randperm(X_aug.size(0))
    return X_aug[perm], y_aug[perm]


## Utils for polygon to grid conversion

## Polygon Refactor to Grid


def _connected_components(free_mask):
    rows, cols = free_mask.shape
    seen = np.zeros_like(free_mask, dtype=bool)
    components = []

    for row in range(rows):
        for col in range(cols):
            if not free_mask[row, col] or seen[row, col]:
                continue
            queue = deque([(row, col)])
            seen[row, col] = True
            component = []

            while queue:
                curr_row, curr_col = queue.popleft()
                component.append((curr_row, curr_col))
                for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_row, next_col = curr_row + d_row, curr_col + d_col
                    if 0 <= next_row < rows and 0 <= next_col < cols and free_mask[next_row, next_col] and not seen[next_row, next_col]:
                        seen[next_row, next_col] = True
                        queue.append((next_row, next_col))

            components.append(component)

    return sorted(components, key=len, reverse=True)

def _shortest_bridge_path(cost_grid, source_cells, target_cells):
    rows, cols = cost_grid.shape
    target_set = set(target_cells)
    heap = []
    best_cost = {}
    parent = {}

    for row, col in source_cells:
        best_cost[(row, col)] = 0.0
        parent[(row, col)] = None
        heappush(heap, (0.0, row, col))

    while heap:
        curr_cost, row, col = heappop(heap)
        if curr_cost > best_cost[(row, col)]:
            continue
        if (row, col) in target_set:
            path = []
            node = (row, col)
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path, curr_cost

        for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_row, next_col = row + d_row, col + d_col
            if not (0 <= next_row < rows and 0 <= next_col < cols):
                continue
            next_cost = curr_cost + cost_grid[next_row, next_col]
            next_node = (next_row, next_col)
            if next_cost < best_cost.get(next_node, float('inf')):
                best_cost[next_node] = next_cost
                parent[next_node] = (row, col)
                heappush(heap, (next_cost, next_row, next_col))

    raise RuntimeError('Could not connect free-space components')

def _enforce_free_space_connectivity(grid, coverage):
    connected_grid = grid.copy()
    free_mask = connected_grid == 0
    components = _connected_components(free_mask)
    if len(components) <= 1:
        return connected_grid

    cost_grid = np.where(connected_grid == 0, 0.0, 1.0 - coverage + 1e-6)

    while len(components) > 1:
        base_component = components[0]
        best_path = None
        best_path_cost = float('inf')

        for other_component in components[1:]:
            path, path_cost = _shortest_bridge_path(cost_grid, base_component, other_component)
            if path_cost < best_path_cost:
                best_path = path
                best_path_cost = path_cost

        for row, col in best_path:
            connected_grid[row, col] = 0
            cost_grid[row, col] = 0.0

        free_mask = connected_grid == 0
        components = _connected_components(free_mask)

    return connected_grid

def polygon_to_obstacle_grid(shell, grid_shape=(64, 64), holes=None, samples_per_cell=4, padding_frac=0.08, ensure_connected=True):
    shell = np.asarray(shell, dtype=float)
    if shell.ndim != 2 or shell.shape[1] != 2:
        raise ValueError('shell must have shape (N, 2)')
    if len(shell) < 3:
        raise ValueError('shell must contain at least 3 vertices')

    holes = [] if holes is None else [np.asarray(h, dtype=float) for h in holes]
    all_xy = np.vstack([shell] + [hole for hole in holes if len(hole) > 0])

    xmin, ymin = all_xy.min(axis=0)
    xmax, ymax = all_xy.max(axis=0)
    width = max(xmax - xmin, 1e-9)
    height = max(ymax - ymin, 1e-9)

    rows, cols = grid_shape
    grid_aspect = cols / rows
    bbox_aspect = width / height

    if bbox_aspect > grid_aspect:
        pad = 0.5 * (width / grid_aspect - height)
        ymin -= pad
        ymax += pad
    else:
        pad = 0.5 * (height * grid_aspect - width)
        xmin -= pad
        xmax += pad

    span_x, span_y = xmax - xmin, ymax - ymin
    xmin -= padding_frac * span_x
    xmax += padding_frac * span_x
    ymin -= padding_frac * span_y
    ymax += padding_frac * span_y

    scale = max(4, samples_per_cell)
    img_w, img_h = cols * scale, rows * scale

    def world_to_px(point):
        px = (point[0] - xmin) / (xmax - xmin) * img_w
        py = (ymax - point[1]) / (ymax - ymin) * img_h
        return (px, py)

    image = Image.new('L', (img_w, img_h), 0)
    draw = ImageDraw.Draw(image)
    draw.polygon([world_to_px(vertex) for vertex in shell], fill=255)

    for hole in holes:
        if len(hole) >= 3:
            draw.polygon([world_to_px(vertex) for vertex in hole], fill=0)

    raster = np.array(image, dtype=float)
    coverage = raster.reshape(rows, scale, cols, scale).mean(axis=(1, 3)) / 255.0
    grid = np.where(coverage >= 0.5, 0, 1).astype(np.uint8)

    if ensure_connected:
        grid = _enforce_free_space_connectivity(grid, coverage)

    extent = (xmin, xmax, ymin, ymax)
    return grid, extent, coverage


def apply_grazing_los(grid, expanded_los):
    """
    Simulates peripheral vision using NumPy array slicing.
    """
    # 1. Identify definitive visible floor (boolean mask)
    visible_floor = (expanded_los == 1) & (grid == 0)
    
    # 2. Initialize empty arrays for the 4 directions
    graze_up    = np.zeros_like(visible_floor)
    graze_down  = np.zeros_like(visible_floor)
    graze_left  = np.zeros_like(visible_floor)
    graze_right = np.zeros_like(visible_floor)
    
    # 3. Shift the visible floor arrays (exactly like our dynamic programming code)
    graze_up[:-1, :]   = visible_floor[1:, :]
    graze_down[1:, :]  = visible_floor[:-1, :]
    graze_left[:, :-1] = visible_floor[:, 1:]
    graze_right[:, 1:] = visible_floor[:, :-1]
    
    # 4. Combine all orthogonal shifts to get the "grazed" area
    grazed_area = visible_floor | graze_up | graze_down | graze_left | graze_right
    
    # 5. The Critical Filter: ONLY keep the grazed cells if they are solid walls
    grazed_walls = grazed_area & (grid == 1)
    
    # 6. Merge the original LOS with the newly illuminated wall boundaries
    final_los = expanded_los | grazed_walls
    
    return final_los

def get_visibility_map_with_LOS(grid, path, grazing_walls, los_type, vision_radius, with_last_obstacle=False):
    unseen_map = np.ones_like(grid, dtype=np.float32)
    agent_position = np.zeros_like(grid, dtype=np.float32)
    
    current_pos = path[-1]
    agent_position[current_pos] = 1.0
    
    if los_type == "los4":
        expanded_los = get_LOS4_visibility_map(grid, path, vision_radius=vision_radius, with_last_obstacle=with_last_obstacle)
    elif los_type == "bresenham":
        expanded_los = get_bresenham_visibility_map(grid, path, vision_radius=vision_radius, with_last_obstacle=with_last_obstacle)
    elif los_type == "los8":
        expanded_los = get_LOS8_visibility_map(grid, path, vision_radius=vision_radius, with_last_obstacle=with_last_obstacle)
    else:
        raise ValueError(f"Unsupported LOS type: {los_type}")

    if grazing_walls:
        expanded_los = apply_grazing_los(grid, expanded_los)
    
    return expanded_los

def generate_training_data_for_online_learning(grid, offline_path, discounted_step = 0, grazing_walls = True, los_type = "los4", vision_radius = float('inf')):
    """
    Generate training data for online learning by simulating the execution of the offline path and recording the state transitions.
    
    Args:
        grid (np.ndarray): The grid representation of the environment.
        offline_path (list of tuples): The path generated by the offline solver.
    Returns:
        3 Channel Tensor:

        Channel 0: 
            Known Obstacles ::
            What it is: The physical walls the agent has actually seen with its Line of Sight (LOS).
            The Math: If a cell is an obstacle AND it has been observed, value = 1.0. Otherwise, value = 0.0.
            Note: At step 0, this channel might be almost entirely zeros, except for the walls immediately surrounding the start position.
        Channel 1: 
            Agent Position ::
            What it is: The exact current location of the agent.
            The Math: A strict one-hot tensor. The single cell where the agent is standing is 1.0. All other 255 cells are 0.0.
        Channel 2: 
            Unseen Map ::
            What it is: The strictly unknown space that the agent's LOS has never touched.
            The Math: If a cell has NEVER been seen, value = 1.0. If a cell HAS been seen (regardless of whether it turned out to be free space or a wall), value = 0.0.
    """

    # Initialize the 3-channel tensor
    X, y = [], []
    
    # Simulate the agent's movement along the offline path
    for step in range(len(offline_path) - 1):
        # Create empty channels
        known_obstacles = np.zeros_like(grid, dtype=np.float32)
        agent_position = np.zeros_like(grid, dtype=np.float32)
        unseen_map = np.ones_like(grid, dtype=np.float32)  # Start with everything unknown
        
        # Get current position of the agent
        current_pos = offline_path[step]
        
        # Update agent position channel (one-hot)
        agent_position[current_pos] = 1.0
        
        # Simulate Line of Sight (LOS) from the current position
        expanded_los = get_visibility_map_with_LOS(grid, offline_path[:step + 1], grazing_walls, los_type, vision_radius, with_last_obstacle=True)
        if grazing_walls:
            expanded_los = apply_grazing_los(grid, expanded_los)

        unseen_map[expanded_los == 1] = 0.0  # This cell has been seen

        # 1. Expand the visible free space by 1 pixel to "touch" the adjacent walls
        # 2. If a cell is touched by the expanded LOS AND it is a wall in the real grid,
        # it is now a Known Obstacle.
        visible_walls = expanded_los & (grid == 1)
        known_obstacles[visible_walls] = 1.0
        
        # Combine channels into a single tensor for this step
        state_tensor = np.stack([known_obstacles, agent_position, unseen_map], axis=0)
        X.append(state_tensor)


        if discounted_step == 0:
            # Original version: Only mark the next step in the offline path as the target
            target_tensor = np.zeros((1, grid.shape[0], grid.shape[1]), dtype=np.float32)
            next_pos = offline_path[step + 1]
            target_tensor[0, next_pos[0], next_pos[1]] = 1.0
            y.append(target_tensor)
        else:
            # ---------------------------------------------------------
            # Discounted Trajectory Heatmap (Comet Tail)
            # ---------------------------------------------------------
            target_tensor = np.zeros((1, grid.shape[0], grid.shape[1]), dtype=np.float32)
            
            gamma = 0.85  # The decay factor (85% strength per step)
            lookahead = discounted_step # How many future steps to paint
            
            for k in range(lookahead):
                future_idx = step + 1 + k
                
                # Ensure we don't look past the end of the expert's path
                if future_idx < len(offline_path):
                    future_pos = offline_path[future_idx]
                    
                    # Calculate the discounted probability
                    discounted_val = gamma ** k
                    
                    # Only overwrite if the new value is higher.
                    # (This prevents a path that loops back on itself from 
                    # overwriting a bright 1.0 with a dim 0.5).
                    if discounted_val > target_tensor[0, future_pos[0], future_pos[1]]:
                        target_tensor[0, future_pos[0], future_pos[1]] = discounted_val
                        
            y.append(target_tensor)
            # ---------------------------------------------------------
    
    return np.array(X), np.array(y)




from data_generator import _solve_grid
def generate_N_training_data_for_online_learning(num_samples, grid_size=(16, 16), density=5, discounted_step = 0, grazing_walls=True, los_type = "los4", vision_radius = float('inf'), timeout=300):
    X_list = []
    y_list = []
    skipped = 0
    
    for _ in tqdm(range(num_samples)):
        # Generate a random grid and path
        gen = WRPDataGenerator(*grid_size)
        # grid, start = gen.generate_valid_grid(density=density)
        grid, start = gen.generate_simple_polygon_grid()
        
        try:
            path_opt, _ = run_with_timeout(_solve_grid, args=(grid, start, los_type, vision_radius), timeout=timeout)
        except TimeoutError:
            skipped += 1
            continue
        
        # Generate training data from the path
        X, y = generate_training_data_for_online_learning(grid, path_opt, discounted_step=discounted_step, grazing_walls=grazing_walls, los_type=los_type, vision_radius=vision_radius)
        X_list.append(torch.tensor(X, dtype=torch.float32))
        y_list.append(torch.tensor(y, dtype=torch.float32))
    
    if skipped:
        print(f"Skipped {skipped}/{num_samples} samples due to timeout ({timeout}s)")
    
    return torch.cat(X_list), torch.cat(y_list)



# How to run
# python data_generator.py --num_samples 1000