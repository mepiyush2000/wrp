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

def get_LOS4_visibility_map(grid, loc_list, with_last_obstacle=False):
        rows, cols = grid.shape
        visibility = np.zeros((rows, cols), dtype=bool)
        
        for loc in loc_list:
            # Check up
            for r in range(loc[0], -1, -1):
                if grid[r, loc[1]] == 1:
                    if with_last_obstacle:
                        visibility[r, loc[1]] = True  # Obstacle is visible
                    break
                visibility[r, loc[1]] = True
            
            # Check down
            for r in range(loc[0], rows):
                if grid[r, loc[1]] == 1:
                    if with_last_obstacle:
                        visibility[r, loc[1]] = True  # Obstacle is visible
                    break
                visibility[r, loc[1]] = True
            
            # Check left
            for c in range(loc[1], -1, -1):
                if grid[loc[0], c] == 1:
                    if with_last_obstacle:
                        visibility[loc[0], c] = True  # Obstacle is visible
                    break
                visibility[loc[0], c] = True
            
            # Check right
            for c in range(loc[1], cols):
                if grid[loc[0], c] == 1:
                    if with_last_obstacle:
                        visibility[loc[0], c] = True  # Obstacle is visible
                    break
                visibility[loc[0], c] = True
            
        return visibility


def greedy_max_visibility_path(grid, start, max_steps=500, verbose=False):
    """Greedy algorithm that always moves to the neighbor that reveals the most unseen cells.
    
    Args:
        grid: numpy array where 0=free, 1=obstacle
        start: tuple (row, col) starting position
        max_steps: maximum number of steps to prevent infinite loops
        verbose: print debug information
        
    Returns:
        path: list of (row, col) tuples representing the path
    """
    H, W = grid.shape
    path = [start]
    current_cell = start
    visited_counts = np.zeros((H, W), dtype=int)
    visited_counts[start[0], start[1]] = 1
    
    # Get initial visibility
    visibility = get_LOS4_visibility_map(grid, path)
    free_space = (grid == 0)
    total_free_cells = np.sum(free_space)
    
    for step in range(max_steps):
        # Check if all free cells are visible
        unseen_free_cells = free_space & (~visibility)
        num_unseen = np.sum(unseen_free_cells)
        
        if num_unseen == 0:
            if verbose:
                print(f"All cells covered in {step} steps! Path length: {len(path)}")
            break
        
        # Evaluate all 4 neighboring directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        best_next_cell = None
        max_new_cells_revealed = -1
        
        r, c = current_cell
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            # Check if the move is valid (within bounds and not an obstacle)
            if not (0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0):
                continue
            
            # Simulate moving to this cell and calculate new visibility
            test_path = path + [(nr, nc)]
            new_visibility = get_LOS4_visibility_map(grid, test_path)
            
            # Count how many NEW cells would be revealed
            newly_revealed = new_visibility & (~visibility)
            num_newly_revealed = np.sum(newly_revealed)
            
            # Add penalty for revisiting cells to encourage exploration
            revisit_penalty = visited_counts[nr, nc] * 0.5
            score = num_newly_revealed - revisit_penalty
            
            if score > max_new_cells_revealed:
                max_new_cells_revealed = score
                best_next_cell = (nr, nc)
        
        # If no valid moves found, we're trapped
        if best_next_cell is None:
            if verbose:
                print(f"Trapped at step {step}! No valid moves. Covered {np.sum(visibility & free_space)}/{total_free_cells} free cells.")
            break
        
        # Make the move
        path.append(best_next_cell)
        current_cell = best_next_cell
        visited_counts[current_cell[0], current_cell[1]] += 1
        
        # Update visibility with the new path
        visibility = get_LOS4_visibility_map(grid, path)
        
        if verbose and step % 10 == 0:
            print(f"Step {step}: at {current_cell}, revealed {np.sum(visibility & free_space)}/{total_free_cells} cells")
    
    return path


def stochastic_visibility_path(grid, start, max_steps=500, temperature=1.0, verbose=False):
    """Stochastic greedy algorithm that samples directions proportional to visibility gains.
    
    Instead of always picking the best direction, this samples from a probability
    distribution where P(direction) ∝ (number of new cells revealed).
    
    Args:
        grid: numpy array where 0=free, 1=obstacle
        start: tuple (row, col) starting position
        max_steps: maximum number of steps
        temperature: controls randomness (higher = more random, lower = more greedy)
        verbose: print debug information
        
    Returns:
        path: list of (row, col) tuples representing the path
    """
    H, W = grid.shape
    path = [start]
    current_cell = start
    visited_counts = np.zeros((H, W), dtype=int)
    visited_counts[start[0], start[1]] = 1
    
    # Get initial visibility
    visibility = get_LOS4_visibility_map(grid, path)
    free_space = (grid == 0)
    total_free_cells = np.sum(free_space)
    
    for step in range(max_steps):
        # Check if all free cells are visible
        unseen_free_cells = free_space & (~visibility)
        num_unseen = np.sum(unseen_free_cells)
        
        if num_unseen == 0:
            if verbose:
                print(f"All cells covered in {step} steps! Path length: {len(path)}")
            break
        
        # Evaluate all 4 neighboring directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        valid_moves = []
        visibility_scores = []
        
        r, c = current_cell
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            # Check if the move is valid
            if not (0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0):
                continue
            
            # Simulate moving to this cell and calculate new visibility
            test_path = path + [(nr, nc)]
            new_visibility = get_LOS4_visibility_map(grid, test_path)
            
            # Count how many NEW cells would be revealed
            newly_revealed = new_visibility & (~visibility)
            num_newly_revealed = np.sum(newly_revealed)
            
            # Score with small penalty for revisiting
            revisit_penalty = visited_counts[nr, nc] * 0.3
            score = max(num_newly_revealed - revisit_penalty, 0.1)  # Minimum score to avoid zero probability
            
            valid_moves.append((nr, nc))
            visibility_scores.append(score)
        
        # If no valid moves, we're trapped
        if not valid_moves:
            if verbose:
                print(f"Trapped at step {step}! No valid moves.")
            break
        
        # Convert scores to probabilities with temperature
        scores_array = np.array(visibility_scores)
        # Apply temperature: lower temperature = more greedy, higher = more random
        exp_scores = np.exp(scores_array / temperature)
        probabilities = exp_scores / np.sum(exp_scores)
        
        # Sample next move based on probabilities
        chosen_idx = np.random.choice(len(valid_moves), p=probabilities)
        best_next_cell = valid_moves[chosen_idx]
        
        # Make the move
        path.append(best_next_cell)
        current_cell = best_next_cell
        visited_counts[current_cell[0], current_cell[1]] += 1
        
        # Update visibility
        visibility = get_LOS4_visibility_map(grid, path)
        
        if verbose and step % 10 == 0:
            print(f"Step {step}: at {current_cell}, revealed {np.sum(visibility & free_space)}/{total_free_cells} cells")
    
    return path


def visibility_guided_search(grid, start, max_expansions=25000, lambda_weight=2.0, verbose=False):
    """A* search guided by visibility gains instead of neural network predictions.
    
    Simplified version that avoids state space explosion by tracking physical steps
    rather than complete visibility sets.
    
    Args:
        grid: numpy array where 0=free, 1=obstacle
        start: tuple (row, col) starting position
        max_expansions: maximum number of nodes to expand
        lambda_weight: weight for the visibility gain in edge cost calculation
        verbose: print debug information
        
    Returns:
        path: list of (row, col) tuples representing the solution path
    """
    import heapq
    
    H, W = grid.shape
    free_space = set((r, c) for r in range(H) for c in range(W) if grid[r, c] == 0)
    
    # Get initial visibility
    init_vis = get_LOS4_visibility_map(grid, [start])
    init_seen = set((r, c) for r in range(H) for c in range(W) if init_vis[r, c] == 1)
    init_unseen = frozenset(free_space - init_seen)
    
    if not init_unseen:
        return [start]
    
    # Priority queue: (f_score, tie_breaker, g, current_cell, path, unseen_set)
    pq = []
    tie_breaker = 0
    heapq.heappush(pq, (0.0, tie_breaker, 0, start, [start], init_unseen))
    
    # Track best g (path length) to each state to avoid re-processing
    visited = {(start, init_unseen): 0}
    
    expansions = 0
    best_path = [start]
    best_unseen_count = len(init_unseen)
    
    while pq:
        f_score, _, g, current_cell, path, unseen_set = heapq.heappop(pq)
        
        # CRITICAL FIX: Skip if we've already processed this state with a better path
        state_key = (current_cell, unseen_set)
        if state_key in visited and visited[state_key] < g:
            continue  # Already found a better path to this state
        
        expansions += 1
        
        # Track best path found so far
        if len(unseen_set) < best_unseen_count:
            best_unseen_count = len(unseen_set)
            best_path = path
            if verbose:
                print(f"Expansion {expansions}: Found path with {len(unseen_set)} unseen cells, path length {g}")
        
        # Check termination conditions
        if expansions > max_expansions:
            # if verbose:
            print(f"Hit max expansions ({max_expansions}). Returning best path with {best_unseen_count} unseen cells.")
            return [start] * 101  # Return dummy path to indicate failure
        
        if len(unseen_set) == 0:
            if verbose:
                print(f"Goal reached! Path length: {g}, Nodes expanded: {expansions}")
            return path
        
        # Get current visibility
        current_vis = get_LOS4_visibility_map(grid, path)
        
        # Explore all 4 directions
        r, c = current_cell
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Calculate visibility gains for all valid moves
        visibility_gains = []
        valid_moves = []
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == 0:
                next_cell = (nr, nc)
                new_path = path + [next_cell]
                
                # Calculate new visibility
                new_vis = get_LOS4_visibility_map(grid, new_path)
                newly_revealed = new_vis & (~current_vis)
                num_newly_revealed = np.sum(newly_revealed)
                
                visibility_gains.append(num_newly_revealed + 1)  # +1 to avoid zero probability
                valid_moves.append((next_cell, new_path, new_vis))
        
        if not valid_moves:
            continue  # No valid moves, backtrack
        
        # Add neighbors to priority queue
        total_gain = sum(visibility_gains)
        for (next_cell, new_path, new_vis), vis_gain in zip(valid_moves, visibility_gains):
            new_g = g + 1
            new_seen = set((vr, vc) for vr in range(H) for vc in range(W) if new_vis[vr, vc] == 1)
            new_unseen = frozenset(free_space - new_seen)
            new_state = (next_cell, new_unseen)
            
            # Only add if we haven't found a better path to this state
            if new_state not in visited or new_g < visited[new_state]:
                visited[new_state] = new_g
                
                # --- NEW FIX: Gravitational Pull to the Fog ---
                if new_unseen:
                    # Find the Manhattan distance to the absolutely closest unseen cell
                    min_dist = min(abs(next_cell[0] - ur) + abs(next_cell[1] - uc) for ur, uc in new_unseen)
                else:
                    min_dist = 0
                
                # 1. Base Heuristic: How much total fog is left?
                h_fog_amount = len(new_unseen) * 0.5 
                
                # 2. Directional Heuristic: How far are we from the nearest fog?
                h_distance = min_dist * 1.0 
                
                # 3. The Immediate Reward: Did we actually reveal anything this step?
                visibility_reward = lambda_weight * np.log(vis_gain)
                
                # 4. Total A* Priority Score
                new_f = new_g + h_fog_amount + h_distance - visibility_reward
                
                tie_breaker += 1
                heapq.heappush(pq, (new_f, tie_breaker, new_g, next_cell, new_path, new_unseen))
    
    if verbose:
        print(f"Search exhausted. Returning best path with {best_unseen_count} unseen cells.")
    return best_path


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



## Data Creators Savers Utils
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


def load_data_from_disk(file_path, sample = 1, device=None):
    """Load dataset tensors saved by save_data_to_disk.

    Args:
        file_path: Path to .pt file
        sample: Fraction of data to load (0 < sample <= 1)
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
    if sample < 1.0:
        num_samples = int(X.size(0) * sample)
        X = X[:num_samples]
        y = y[:num_samples] 

    print(f"Loaded {X.size(0)} samples from {file_path}")
    print(f"X shape: {tuple(X.shape)} | y shape: {tuple(y.shape)}")
    return X, y


def save_evaluation_results(gt_lengths, pred_lengths_model, pred_lengths_search, pred_lengths_visibility, filename="evaluation_results.npz"):
    np.savez(filename, gt_lengths=gt_lengths, pred_lengths_model=pred_lengths_model, pred_lengths_search=pred_lengths_search, pred_lengths_visibility=pred_lengths_visibility)
    print(f"Saved evaluation results to {filename}")
def load_evaluation_results(filename="evaluation_results.npz"):
    data = np.load(filename)
    gt_lengths = data['gt_lengths']
    pred_lengths_model = data['pred_lengths_model']
    pred_lengths_search = data['pred_lengths_search']
    pred_lengths_visibility = data['pred_lengths_visibility']
    print(f"Loaded evaluation results from {filename}")
    return gt_lengths, pred_lengths_model, pred_lengths_search, pred_lengths_visibility


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



