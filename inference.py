import numpy as np
import torch
import heapq
from utils import get_visibility_map_with_LOS


@torch.no_grad()
def generate_flow_heatmap(model, context_tensor, device, inference_steps=10, cfg_scale=1.0):
    """
    Solves the Rectified Flow ODE to generate the final path heatmap.
    inference_steps: How many times to query the U-Net. 5 is usually plenty for straight-line ODEs!
    """
    context_tensor = context_tensor.to(device)
    empty_context = torch.zeros_like(context_tensor)
    b, c, h, w = context_tensor.shape
    
    # 1. Start with pure random noise (x_0)
    x_t = torch.randn((b, 1, h, w), device=device)
    
    # The size of our time step
    dt = 1.0 / inference_steps
    
    # 2. Iteratively sculpt the noise into the path
    for i in range(inference_steps):
        # Current time t
        t_val = i / inference_steps
        t_tensor = torch.full((b,), t_val, device=device)
        
        with torch.no_grad():
            # Pass 1: Conditional (The model sees the maze walls)
            v_cond = model(context_tensor, noisy_path=x_t, t=t_tensor)
            
            # Pass 2: Unconditional (The model sees nothing)
            if cfg_scale > 1.0:
                v_uncond = model(empty_context, noisy_path=x_t, t=t_tensor)
                velocity = v_uncond + cfg_scale * (v_cond - v_uncond)
                
            else:
                velocity = v_cond
            
        # Euler Step
        x_t = x_t + (velocity * dt)
        # plot_output_tensor(x_t[0])

        
    # After the loop, x_t has reached t=1.0 and is our final sculpted heatmap!
    return x_t


NEIGHBOURS = [(0, 1), (0, -1), (1, 0), (-1, 0)] #Agent is designed to move only in four cardinal direction for this experiment

def local_astar(start, goal, known_obstacles):
    """
    Standard A* that ONLY routes through known free space.
    """
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path # Returns the path EXCLUDING the start node
            
        for dr, dc in NEIGHBOURS:
            neighbor = (current[0] + dr, current[1] + dc)
            
            if 0 <= neighbor[0] < known_obstacles.shape[0] and 0 <= neighbor[1] < known_obstacles.shape[1]:
                if known_obstacles[neighbor[0], neighbor[1]] == 1.0:
                    continue
                    
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
    return None

def find_all_frontiers(known_obstacles, unseen_map, H, W):
    frontiers = []
    for r in range(H):
        for c in range(W):
            if known_obstacles[r, c] == 0.0 and unseen_map[r, c] == 0.0: 
                for dr, dc in NEIGHBOURS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and unseen_map[nr, nc] == 1.0:
                        frontiers.append((r, c))
                        break # Break the inner directional loop, move to next cell
    return frontiers


def execute_frontier_escape(current_pos, frontiers, pred, known_obstacles, grid, path, 
                            visited_counts, state_tensor, 
                            los_type, vision_radius, dataX = None, dataY =None, verbose=True):
    
    best_frontier = max(frontiers, key=lambda f: pred[f[0], f[1]])
    escape_path = local_astar(current_pos, best_frontier, known_obstacles)
    escaped = False

    unseen_map = state_tensor[0, 2].cpu().numpy()
    
    if escape_path and len(escape_path) > 0:
        
        # --- THE FIX: DYNAMIC START NODE CHECK ---
        # If your A* includes the agent's current position at index 0, remove it.
        # If it only returns future steps, leave it alone.
        if escape_path[0] == current_pos:
            actual_steps = escape_path[1:]
        else:
            actual_steps = escape_path
        # -----------------------------------------
            
        if len(actual_steps) > 0:
                
            for next_step in actual_steps:
                path.append(next_step)
                current_pos = next_step
                visited_counts[current_pos[0], current_pos[1]] += 1
                
                if dataX is not None and dataY is not None:
                    dataX.append(state_tensor.cpu().numpy())
                    dataY.append(pred)
                
                current_los = get_visibility_map_with_LOS(
                    grid, path, 
                    grazing_walls=True, 
                    los_type=los_type, 
                    vision_radius=vision_radius, 
                    with_last_obstacle=True
                )
                
                revealed_new_fog = np.any((unseen_map == 1.0) & (current_los == 1))
                
                if revealed_new_fog:
                    if verbose:
                        print(f"New fog revealed! Breaking escape to replan.")
                    break
                    
            escaped = True
        
    return current_pos, escaped
N_RUN = 8

def pred_path_greedy_online_flow(model, grid, start, los_type, vision_radius, device, max_steps=180, verbose=True, stop_at_offline_compare = False):
    model.eval()
    current_pos = start
    path = [current_pos]
    H, W = grid.shape
    visited_counts = np.zeros((H, W))
    visited_counts[start[0], start[1]] = 1
    dataX =[]
    dataY =[]
    total_floor_cells = np.sum(grid == 0)
    
    for step_idx in range(max_steps):
        # 1. Create input tensor for the current state (Unchanged)
        known_obstacles = np.zeros_like(grid, dtype=np.float32)
        agent_position = np.zeros_like(grid, dtype=np.float32)
        unseen_map = np.ones_like(grid, dtype=np.float32)  
        
        agent_position[current_pos] = 1.0
        
        grazing_los = get_visibility_map_with_LOS(grid, path, grazing_walls=True, los_type=los_type, vision_radius=vision_radius, with_last_obstacle=True)
        unseen_map[grazing_los == 1] = 0.0
        visible_walls = (grazing_los == 1) & (grid == 1)
        known_obstacles[visible_walls] = 1.0

        if stop_at_offline_compare:
            # 2. Your Exact Termination Condition
            explored_floor_cells = np.sum((unseen_map == 0.0) & (known_obstacles == 0.0))
            
            if explored_floor_cells == total_floor_cells:
                if verbose:
                    print(f"Success! Entire grid explored optimally in {step_idx} steps.")
                break


        # --- MODIFIED: Gather ALL Frontiers ---
        frontiers = find_all_frontiers(known_obstacles, unseen_map, H, W)

                            
        if not frontiers:
            if verbose:
                print(f"Success! Entire polygon explored in {len(path)-1} steps.")
            break


        # 2. Context Preparation
        state_tensor = np.stack([known_obstacles, agent_position, unseen_map], axis=0)
        state_tensor = torch.tensor(state_tensor, dtype=torch.float32).unsqueeze(0).to(device)

        # 3. THE FLOW MATCHING UPGRADE
        # Generate the heatmap by solving the ODE
        
        state_tensor_batched = state_tensor.repeat(N_RUN, 1, 1, 1)
        samples = generate_flow_heatmap(model, state_tensor_batched, device=device)
        pred = (samples.mean(dim=0)).cpu().numpy()[0]  # Average over runs and remove batch dimension
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)  # Normalize to [0, 1]

        
        # -----------------------------------------------------
        # 3. LOOP DETECTION & MODULAR ESCAPE
        # -----------------------------------------------------
        if visited_counts[current_pos[0], current_pos[1]] >= 3 or len(frontiers) ==1:
            if verbose: 
                if len(frontiers) == 1:
                    print(f"Few frontiers ({len(frontiers)}) left at step {step_idx} and pos {current_pos}. Triggering A* Escape.")
                else:
                    print(f"Loop detected at step {step_idx} and pos {current_pos}! Triggering A* Escape.")
            
            # Call our new modular function!
            current_pos, escaped = execute_frontier_escape(
                current_pos, frontiers, pred, known_obstacles, grid, path, 
                visited_counts, state_tensor, 
                los_type, vision_radius, dataX, dataY,verbose
            )
            
            if escaped:
                if verbose: print(f"Escape successful! New position: {current_pos}. Continuing exploration.")
                continue # Restart the main loop from the newly escaped position!
        # -----------------------------------------------------

        # 4. Physically Constrained Action Selection
        neighbors = [
            (current_pos[0]-1, current_pos[1]), # Up
            (current_pos[0]+1, current_pos[1]), # Down
            (current_pos[0], current_pos[1]-1), # Left
            (current_pos[0], current_pos[1]+1)  # Right
        ]
        
        best_prob = -float('inf')
        best_next_pos = current_pos
        
        for r, c in neighbors:
            if 0 <= r < H and 0 <= c < W:
                if known_obstacles[r, c] != 1.0: # Don't walk into known walls
                    # Because we trained on a discounted 10-step horizon,
                    # the immediate correct step will naturally be the brightest pixel!
                    prob = pred[r, c]
                    penalty = visited_counts[r, c] * 0.4  # Keeps the agent from stalling
                    score = prob - penalty
                    
                    if score > best_prob:
                        best_prob = score
                        best_next_pos = (r, c)
                        
        if best_next_pos == current_pos:
            print(f"Model predicted to stay in place at step {step_idx}. Ending path.")
            break
        
        path.append(best_next_pos)
        current_pos = best_next_pos
        visited_counts[current_pos[0], current_pos[1]] += 1
        dataX.append(state_tensor.cpu().numpy())
        dataY.append(pred)
    return path, (dataX, dataY)




def receding_horizon_astar_online(model, grid, start, los_type, vision_radius , device, max_steps=180, verbose = True, stop_at_offline_compare = False):
    model.eval()
    current_pos = start
    path = [current_pos]
    H, W = grid.shape
    
    # --- THE FIX: MEMORY ARRAYS FOR PLAN COMMITMENT ---
    current_plan = []
    previous_unseen_count = float('inf')
    total_floor_cells = np.sum(grid == 0)

    # --------------------------------------------------
    
    for step in range(max_steps):
        # 1. State Construction
        agent_position = np.zeros_like(grid, dtype=np.float32)
        known_obstacles = np.zeros_like(grid, dtype=np.float32)
        unseen_map = np.ones_like(grid, dtype=np.float32)
        agent_position[current_pos] = 1.0
        
        grazing_los = get_visibility_map_with_LOS(grid, path, grazing_walls=True, los_type=los_type, vision_radius=vision_radius, with_last_obstacle=True)
        unseen_map[grazing_los == 1] = 0.0
        visible_walls = (grazing_los==1) & (grid == 1)
        known_obstacles[visible_walls] = 1.0

        if stop_at_offline_compare:
            # 2. Your Exact Termination Condition
            explored_floor_cells = np.sum((unseen_map == 0.0) & (known_obstacles == 0.0))
            
            if explored_floor_cells == total_floor_cells:
                if verbose:
                    print(f"Success! Entire grid explored optimally in {step} steps.")
                break
            
        current_unseen_count = unseen_map.sum()

        # ---------------------------------------------------------
        # EVENT-DRIVEN REPLANNING
        # We ONLY run the U-Net and A* if we have no plan, OR if the fog changed.
        # ---------------------------------------------------------
        
        if not current_plan or current_unseen_count < previous_unseen_count:
            
            # Update our tracker
            previous_unseen_count = current_unseen_count
            
            # 3. Get U-Net Prediction
            state_tensor = np.stack([known_obstacles, agent_position, unseen_map], axis=0)
            state_tensor = torch.tensor(state_tensor, dtype=torch.float32).unsqueeze(0).to(device)

            state_tensor_batched = state_tensor.repeat(N_RUN, 1, 1, 1)
            samples = generate_flow_heatmap(model, state_tensor_batched, device=device)
            pred = (samples.mean(dim=0)).cpu().numpy()[0]  # Average over runs and remove batch dimension
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)  # Normalize to [0, 1]
                
            # 4. Find all active Frontiers
            frontiers = find_all_frontiers(known_obstacles, unseen_map, H, W)
                                
            if not frontiers:
                if verbose:
                    print(f"Success! Entire polygon explored in {step} steps.")
                break

            # 5. Score Frontiers and COMMIT to the best path
            best_path = None
            best_score = -float('inf')
            
            for f in frontiers:
                if f == current_pos:
                    continue

                # --- NEW: Information Gain (Guaranteed Area) ---
                # Dynamically simulate vision based on the active LOS settings
                sim_los = get_visibility_map_with_LOS(known_obstacles, [f], grazing_walls=True, los_type=los_type, vision_radius=vision_radius, with_last_obstacle=False)

                # The gain is strictly the overlap between what the agent *would* see 
                # and the fog that is *currently* unseen.
                simulated_gain = np.sum((sim_los == True) & (unseen_map == 1.0))
                # ------------------------------------------------------
                if simulated_gain == 0:
                    continue  # Skip frontiers that provide no new information
                    
                local_path = local_astar(current_pos, f, known_obstacles)
                
                if local_path is not None and len(local_path) > 0:
                    first_step = local_path[0]
                    prob = pred[first_step[0], first_step[1]]
                    distance = len(local_path)
                    
                    # TWEAK: Increased the distance penalty from 0.01 to 0.05
                    # This stops the agent from walking across the map for a 1% probability gain
                    # score = prob - 0.05 * distance #prob * simulated_gain/distance 

                    path_prob_sum = sum(pred[r, c] for r, c in local_path if pred[r, c] > 0)  # Only sum positive probabilities
                    avg_path_prob = path_prob_sum / distance
                    normalized_dist = distance / (H+W)
                   
                    # Strictly Multiplicative Fusion
                    score = avg_path_prob * np.exp(-3 * normalized_dist) # * decay_penalty
                    # score = avg_path_prob - 0.25 * distance
                    # print("Prob sum:", path_prob_sum, "Normalized dist:", normalized_dist, "Score:", score)
                    
                    if score > best_score:
                        best_score = score
                        best_path = local_path
                        
            if best_path is None:
                print(f"Agent trapped at step {step}. Frontiers exist but are walled off.")
                break
                
            # SAVE THE ENTIRE PATH TO MEMORY
            current_plan = best_path
            
        # ---------------------------------------------------------
        # THE EXECUTION
        # ---------------------------------------------------------
        # Pop the next step off the plan and physically take it
        next_step = current_plan.pop(0)
        path.append(next_step)
        current_pos = next_step
        
    return path



def neasrest_frontier_algorithm(grid, start, los_type, vision_radius, max_steps=180, verbose = True, stop_at_offline_compare = False):
    current_pos = start
    path = [current_pos]
    
    # --- THE FIX: MEMORY ARRAYS FOR PLAN COMMITMENT ---
    current_plan = []
    previous_unseen_count = float('inf')
    total_floor_cells = np.sum(grid == 0)

    # --------------------------------------------------
    
    for step in range(max_steps):
        # 1. State Construction
        agent_position = np.zeros_like(grid, dtype=np.float32)
        known_obstacles = np.zeros_like(grid, dtype=np.float32)
        unseen_map = np.ones_like(grid, dtype=np.float32)
        agent_position[current_pos] = 1.0
        
        grazing_los = get_visibility_map_with_LOS(grid, path, grazing_walls=True, los_type=los_type, vision_radius=vision_radius, with_last_obstacle=True)
        unseen_map[grazing_los == 1] = 0.0
        visible_walls = grazing_los & (grid == 1)
        known_obstacles[visible_walls] = 1.0

        if stop_at_offline_compare:
            # 2. Your Exact Termination Condition
            explored_floor_cells = np.sum((unseen_map == 0.0) & (known_obstacles == 0.0))
            
            if explored_floor_cells == total_floor_cells:
                if verbose:
                    print(f"Success! Entire grid explored optimally in {step} steps.")
                break

        current_unseen_count = unseen_map.sum()

        # ---------------------------------------------------------
        # EVENT-DRIVEN REPLANNING
        # We ONLY run the U-Net and A* if we have no plan, OR if the fog changed.
        # ---------------------------------------------------------
        if not current_plan or current_unseen_count < previous_unseen_count:
            
            # Update our tracker
            previous_unseen_count = current_unseen_count
            
            state_tensor = np.stack([known_obstacles, agent_position, unseen_map], axis=0)
            state_tensor = torch.tensor(state_tensor, dtype=torch.float32).unsqueeze(0)

                
            # 4. Find all active Frontiers
            frontiers = []
            H, W = grid.shape
            frontiers = find_all_frontiers(known_obstacles, unseen_map, H, W)
            if not frontiers:
                if verbose:
                    print(f"Success! Entire polygon explored in {step} steps.")
                break

            # 5. Score Frontiers and COMMIT to the best path
            best_path = None
            best_score = -float('inf')
            
            for f in frontiers:
                if f == current_pos:
                    continue
                    
                 # --- NEW: Information Gain (Guaranteed Area) ---
                # Dynamically simulate vision based on the active LOS settings
                sim_los = get_visibility_map_with_LOS(known_obstacles, [f], grazing_walls=True, los_type=los_type, vision_radius=vision_radius, with_last_obstacle=False)

                # The gain is strictly the overlap between what the agent *would* see 
                # and the fog that is *currently* unseen.
                simulated_gain = np.sum((sim_los == True) & (unseen_map == 1.0))
                # ------------------------------------------------------

                if simulated_gain == 0:
                    continue  # Skip frontiers that provide no new information

                local_path = local_astar(current_pos, f, known_obstacles)
                
                if local_path is not None and len(local_path) > 0:
                    first_step = local_path[0]
                    # prob = pred[first_step[0], first_step[1]]
                    distance = len(local_path)
                    
                    # --- THE NEW IPP HEURISTIC ---
                    # Weight the U-Net's probability by the immediate mapping gain,
                    # then apply the linear physical distance tax.
                    score = (1) / ( distance )
                    # -----------------------------
                    
                    if score > best_score:
                        best_score = score
                        best_path = local_path
                        
            if best_path is None:
                print(f"Agent trapped at step {step}. Frontiers exist but are walled off.")
                break
                
            # SAVE THE ENTIRE PATH TO MEMORY
            current_plan = best_path
            
        # ---------------------------------------------------------
        # THE EXECUTION
        # ---------------------------------------------------------
        # Pop the next step off the plan and physically take it
        next_step = current_plan.pop(0)
        path.append(next_step)
        current_pos = next_step
        
    return path


import math

def actual_path_length(coords):
    """
    Distance covered along a path of adjacent (row, col) cells (8-connected).
    Each cardinal step = 1, each diagonal step = sqrt(2).
    """
    total = 0.0
    for (r1, c1), (r2, c2) in zip(coords[:-1], coords[1:]):
        dr, dc = abs(r2 - r1), abs(c2 - c1)
        if dr == 1 and dc == 1:
            total += math.sqrt(2)      # diagonal
        else:
            total += 1.0               # cardinal (dr+dc == 1)
    return round(total, 2)


# ---------------------- Helper function ----------------------
def plot_percentage_hist(ax, data, step,  title, color="skyblue", ratio_norm = False, title_size=15, label_size=16, tick_size=16, legend_size=16):
    weights = np.ones_like(data, dtype=float) * 100.0 / len(data)
    bins = np.arange(0.75, 3 + step, step)

    data_ratio= [1 if x < 1 else x for x in data]

    ax.hist(
        data_ratio if ratio_norm else data,
        bins=bins,
        weights=weights,
        edgecolor="black",
        color=color
    )
    ax.set_title(title, fontsize=title_size, fontweight="bold")
    ax.set_xlabel("Path Length Ratio", fontsize=label_size)
    ax.set_ylabel("Percentage (%)", fontsize=label_size)
    ax.axvline(1, color="red", linestyle="dashed", linewidth=2)

    mean_ratio = np.mean(data)
    ax.axvline(
        mean_ratio,
        color="green",
        linestyle="dashed",
        linewidth=2,
        label=f"MCR: {mean_ratio:.2f}"
    )

    ax.set_xticks(np.arange(0.75, 3 + step, step))
    ax.set_yticks(np.arange(0, 65, 10))
    ax.tick_params(axis="x", rotation=90, labelsize=tick_size)
    ax.tick_params(axis="y", labelsize=tick_size)
    ax.legend(fontsize=legend_size)
    ax.grid(alpha=0.3)

# ---------------------- Create plots ----------------------