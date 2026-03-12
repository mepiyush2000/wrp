import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import multiprocessing as _mp
import traceback as _tb


def _worker(fn, args, kwargs, result_queue):
    """Target for the child process."""
    try:
        result = fn(*args, **kwargs)
        result_queue.put(("ok", result))
    except Exception as e:
        # Send traceback string since the original exception may not pickle
        result_queue.put(("err", f"{type(e).__name__}: {e}\n{''.join(_tb.format_exception(type(e), e, e.__traceback__))}"))


def run_with_timeout(fn, args=(), kwargs=None, timeout=10):
    """Run a function with a timeout (in seconds).

    Uses a separate *process* so the work is truly killed on timeout
    (threads cannot be stopped in Python).

    IMPORTANT: ``fn`` must be defined in an importable module (not in a
    notebook cell) because Windows uses the 'spawn' multiprocessing start
    method, which needs to pickle and re-import the target function.

    Returns the function's result on success, or raises TimeoutError.

    Usage:
        from wrp_solver_opt import solve_grid
        path, cost = run_with_timeout(solve_grid, args=(grid, start), timeout=30)
    """
    if kwargs is None:
        kwargs = {}
    ctx = _mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(fn, args, kwargs, q))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(f"Function {fn.__name__} timed out after {timeout}s")

    if q.empty():
        raise RuntimeError(
            f"Child process for {fn.__name__} exited with code {p.exitcode} "
            f"but returned no result. If the function is defined in a notebook "
            f"cell, move it to an importable .py module."
        )

    status, payload = q.get_nowait()
    if status == "ok":
        return payload
    raise RuntimeError(payload)

def apply_spatial_smoothing(grid_tensor, target_map, smooth_val=0.2):
    """
    Applies spatial label smoothing to adjacent free cells.
    target_map: (H, W) tensor with 1.0 at the target location.
    grid_tensor: (H, W) tensor where 1.0 is an obstacle and 0.0 is free space.
    """
    smoothed_map = target_map.clone()
    
    # Find the coordinates of the target cell (where value is 1.0)
    target_indices = torch.nonzero(target_map == 1.0)
    
    # Safety check in case of an empty target map
    if len(target_indices) == 0:
        return smoothed_map
        
    r, c = target_indices[0]
    
    # 4-way neighbors (Up, Down, Left, Right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    H, W = target_map.shape
    
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        
        # 1. Check if neighbor is within grid bounds
        if 0 <= nr < H and 0 <= nc < W:
            # 2. Check if neighbor is free space (not a wall)
            if grid_tensor[nr, nc] == 0.0:
                # Assign the smoothed probability
                smoothed_map[nr, nc] = smooth_val
                
    return smoothed_map

def get_LOS4_visibility_map(grid, loc_list):
        rows, cols = grid.shape
        visibility = np.zeros((rows, cols), dtype=bool)
        
        for loc in loc_list:
            # Check up
            for r in range(loc[0], -1, -1):
                if grid[r, loc[1]] == 1:
                    break
                visibility[r, loc[1]] = True
            
            # Check down
            for r in range(loc[0], rows):
                if grid[r, loc[1]] == 1:
                    break
                visibility[r, loc[1]] = True
            
            # Check left
            for c in range(loc[1], -1, -1):
                if grid[loc[0], c] == 1:
                    break
                visibility[loc[0], c] = True
            
            # Check right
            for c in range(loc[1], cols):
                if grid[loc[0], c] == 1:
                    break
                visibility[loc[0], c] = True
            
        return visibility

def get_path_from_y_labels(y_labels):
    """Converts a sequence of one-hot encoded maps into a path."""
    path = []
    for y in y_labels:
        # y shape: (1, H, W)
        y = y.squeeze(0)  # Shape: (H, W)
        next_cell_idx = torch.argmax(y).item()  # Get the index of the max value
        H, W = y.shape
        row = next_cell_idx // W
        col = next_cell_idx % W
        path.append((row, col))
    return path


def save_data_to_disk(x, y, file_path="wrp_dataset.pt", to_cpu=True):
    """Save dataset tensors to disk using torch.save.

    Args:
        x: Input tensor (N, C, H, W)
        y: Label tensor (N, 1, H, W)
        file_path: Output .pt file path
        to_cpu: If True, move tensors to CPU before saving
    """
    if not torch.is_tensor(x) or not torch.is_tensor(y):
        raise TypeError("x and y must be torch tensors")
    if x.size(0) != y.size(0):
        raise ValueError("x and y must have the same number of samples")

    x_save = x.detach().clone().contiguous()
    y_save = y.detach().clone().contiguous()

    if to_cpu:
        x_save = x_save.cpu()
        y_save = y_save.cpu()

    payload = {
        "X": x_save,
        "y": y_save,
        "num_samples": x_save.size(0),
        "x_shape": tuple(x_save.shape),
        "y_shape": tuple(y_save.shape)
    }

    os.makedirs(os.path.dirname(file_path), exist_ok=True) if os.path.dirname(file_path) else None
    torch.save(payload, file_path)

    print(f"Saved {payload['num_samples']} samples to {file_path}")
    print(f"X shape: {payload['x_shape']} | y shape: {payload['y_shape']}")
    return file_path


def load_data_from_disk(file_path, device=None):
    """Load dataset tensors saved by save_data_to_disk.

    Args:
        file_path: Path to .pt file
        device: Optional device (e.g., DEVICE) to move tensors
    Returns:
        X, y tensors
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    payload = torch.load(file_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Invalid file format: expected dictionary payload")
    if "X" not in payload or "y" not in payload:
        raise KeyError("Invalid payload: keys 'X' and 'y' are required")

    X = payload["X"]
    y = payload["y"]

    if not torch.is_tensor(X) or not torch.is_tensor(y):
        raise TypeError("Invalid payload: 'X' and 'y' must be tensors")
    if X.size(0) != y.size(0):
        raise ValueError("Corrupt payload: sample count mismatch between X and y")

    if device is not None:
        X = X.to(device)
        y = y.to(device)

    print(f"Loaded {X.size(0)} samples from {file_path}")
    print(f"X shape: {tuple(X.shape)} | y shape: {tuple(y.shape)}")
    return X, y

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


# def remove_spatial_smoothening_from_the_data(data_train):
#     """Convert smoothed label maps to hard one-hot maps.

#     Expected shape:
#       - (N, 1, H, W) or (N, H, W)
#     Returns:
#       - Same shape as input with exactly one 1.0 per sample map.
#     """
#     if not torch.is_tensor(data_train):
#         raise TypeError("data_train must be a torch.Tensor")

#     original_shape = data_train.shape
#     squeeze_channel = False

#     if data_train.ndim == 4:
#         if data_train.size(1) != 1:
#             raise ValueError("For 4D input, expected shape (N, 1, H, W)")
#         maps = data_train.squeeze(1)  # (N, H, W)
#         squeeze_channel = True
#     elif data_train.ndim == 3:
#         maps = data_train  # (N, H, W)
#     else:
#         raise ValueError("Expected input shape (N, 1, H, W) or (N, H, W)")

#     n, h, w = maps.shape
#     flat = maps.view(n, -1)
#     max_idx = flat.argmax(dim=1)

#     hard_flat = torch.zeros_like(flat)
#     hard_flat[torch.arange(n, device=maps.device), max_idx] = 1.0
#     hard_maps = hard_flat.view(n, h, w)

#     if squeeze_channel:
#         hard_maps = hard_maps.unsqueeze(1)  # (N, 1, H, W)

#     if hard_maps.shape != original_shape:
#         raise RuntimeError("Output shape mismatch after removing smoothening")

#     return hard_maps

# # Example usage:
# # y_train_hard = remove_spatial_smoothening_from_the_data(y_train)
# y_train_aug = remove_spatial_smoothening_from_the_data(y_train_aug)



## Plot Utils




def plot_grid(grid, start):
    plt.imshow(1 - grid, cmap='gray')
    if start is not None:
        plt.plot(start[1], start[0], 'go')
    plt.grid(True)
    plt.xticks(np.arange(-0.5, grid.shape[1], 1))
    plt.yticks(np.arange(-0.5, grid.shape[0], 1))
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.show()


def plot_path(grid, path, start=None):
    plt.imshow(1 - grid, cmap='gray')
    path = np.array(path)
    plt.plot(path[:, 1], path[:, 0], 'b--')
    plt.grid(True)

    if start is not None:
        plt.plot(start[1], start[0], 'go')
    else:
        plt.plot(path[0][1], path[0][0], 'go')

    plt.plot(path[-1][1], path[-1][0], 'ro')
    plt.xticks(np.arange(-0.5, grid.shape[1], 1))
    plt.yticks(np.arange(-0.5, grid.shape[0], 1))
    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.show()


def plot_visibility(grid, path, unseen=False):
    visibility = get_LOS4_visibility_map(grid, path)

    if unseen:
        plt.imshow((1 - visibility) * (1 - grid), cmap='gray')
    else:
        plt.imshow(visibility, cmap='gray')

    plt.grid(True)
    plt.xticks(np.arange(-0.5, grid.shape[1], 1))
    plt.yticks(np.arange(-0.5, grid.shape[0], 1))

    path_arr = np.array(path)
    plt.plot(path_arr[:, 1], path_arr[:, 0], 'b-', linewidth=1, alpha=0.6)
    plt.plot(path[0][1], path[0][0], 'go', markersize=8, label='Start')
    plt.plot(path[-1][1], path[-1][0], 'ro', markersize=8, label='End')

    plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.title('Cumulative LOS4 Visibility Along Path')
    plt.show()


def plot_output_tensor(output_tensor):
    output_tensor = output_tensor.squeeze(0).squeeze(0)
    plt.imshow(output_tensor.detach().cpu().numpy(), cmap='hot')
    plt.colorbar()
    plt.grid()
    plt.xticks(np.arange(-0.5, output_tensor.shape[1], 1))
    plt.yticks(np.arange(-0.5, output_tensor.shape[0], 1))
    plt.title('Model Output Heatmap')
    plt.show()
