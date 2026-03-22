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



## Utils for polygon to grid conversion
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