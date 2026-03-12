import torch
from tqdm import tqdm
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

def _solve_grid(grid, start):
    solver = WRPSolverTSPJF(grid, start)
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