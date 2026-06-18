import json

import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import torch
from wrp_solver_suboptimal import *
from data_generator import generate_training_data_for_online_learning
from utils import *

def image_to_grid(path, rows=24, cols=32, light_thresh=152):
    """
    Convert the grid-map image into a (rows x cols) array — EXACT at native 40x30.
      light grey -> 1 (free), dark grey -> 0 (background), yellow -> start (marked 1).
    Returns (grid, start) with start a (row, col) tuple, or None.
    """
    img = np.array(Image.open(path).convert("RGB")).astype(int)
    H, W, _ = img.shape
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    yellow = (R > 150) & (G > 150) & (B < 100)
    start = None
    if yellow.any():
        ys, xs = np.where(yellow)
        start = (int(round(ys.mean() * rows / H)),
                 int(round(xs.mean() * cols / W)))

    grid = np.zeros((rows, cols), dtype=np.int8)
    for r in range(rows):
        for c in range(cols):
            y0, y1 = int(r * H / rows), int((r + 1) * H / rows)
            x0, x1 = int(c * W / cols), int((c + 1) * W / cols)
            block_is_light = (img[y0:y1, x0:x1].mean(axis=2) > light_thresh).mean() > 0.5
            grid[r, c] = 1 if block_is_light else 0

    if start is not None:
        grid[start] = 1

    grid = np.vstack([np.array([[0] * 32] * 4), grid, np.array([[0] * 32] * 4)])
    start = (start[0] + 4, start[1])
    return 1-grid, start


def image_to_grid2(path, mode="anyfree", light_thresh=152):
    """
    Convert the map image to a 24x32 grid (rows x cols). and then to 32x32 by padding 4 rows of walls on top and bottom.
      light grey -> 1 (free), dark grey -> 0, yellow -> start (marked 1).
    mode="majority": faithful area rounding (tighter borders, may thin 1-cell corridors)
    mode="anyfree" : keep a cell free if any corridor overlaps it (preserves all passages)
    Returns (grid, start).
    """
    img = np.array(Image.open(path).convert("RGB")).astype(int)
    H, W, _ = img.shape
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # 1) exact native 40x30 grid (aligns with the map's true 16px cells)
    NR, NC = 30, 40
    native = np.zeros((NR, NC), np.int8)
    for r in range(NR):
        for c in range(NC):
            y0, y1 = int(r*H/NR), int((r+1)*H/NR)
            x0, x1 = int(c*W/NC), int((c+1)*W/NC)
            native[r, c] = 1 if (img[y0:y1, x0:x1].mean(2) > light_thresh).mean() > 0.5 else 0

    # 2) downsample 40x30 -> 24x32 with the chosen rounding rule
    rows, cols = 24, 32
    grid = np.zeros((rows, cols), np.int8)
    for r in range(rows):
        for c in range(cols):
            y0, y1 = int(r*NR/rows), int((r+1)*NR/rows)
            x0, x1 = int(c*NC/cols), int((c+1)*NC/cols)
            blk = native[y0:y1, x0:x1]
            grid[r, c] = (blk.mean() > 0.5) if mode == "majority" else int(blk.any())

    # 3) start from the yellow centroid
    yellow = (R > 150) & (G > 150) & (B < 100)
    start = None
    if yellow.any():
        ys, xs = np.where(yellow)
        start = (int(round(ys.mean()*rows/H)), int(round(xs.mean()*cols/W)))
        grid[start] = 1

    grid = np.vstack([np.array([[0] * 32] * 4), grid, np.array([[0] * 32] * 4)])
    start = (start[0] + 4, start[1])
    return 1-grid, start

def _solve_grid_subopt(grid, start, los_type = "los4", vision_radius = float('inf')):
    solver = WRPSolverJF(grid, start, los_type=los_type, vision_radius=vision_radius)
    return solve_wrp_jf(solver, weight=1, df=6, heuristic="tsp")

def generate_N_training_data_for_online_learning_from_folder(folder_path, discounted_step = 0, grazing_walls=True, los_type = "los4", vision_radius = float('inf'), timeout=600):
    X_list = []
    y_list = []
    skipped = 0
    
    folder_paths = sorted(os.listdir(folder_path))[:1500]
    for imname in tqdm(folder_paths):
        if (imname[-4:]!=".png"): continue
        impath = os.path.join(folder_path, imname)
        # Generate a random grid and path
        grid, start = image_to_grid2(impath)
        # grid, start = gen.generate_simple_polygon_grid()
        
        try:
            path_opt, _ = run_with_timeout(_solve_grid_subopt, args=(grid, start, los_type, vision_radius), timeout=timeout)
            resJson = {"grid": grid, "start": start, "path_opt": path_opt}
            resJsonPath = impath.replace(".png", ".npy")
            np.save(resJsonPath, resJson)

        except TimeoutError:
            skipped += 1
            continue
        
        # Generate training data from the path
        X, y = generate_training_data_for_online_learning(grid, path_opt, discounted_step=discounted_step, grazing_walls=grazing_walls, los_type=los_type, vision_radius=vision_radius)
        X_list.append(torch.tensor(X, dtype=torch.float32))
        y_list.append(torch.tensor(y, dtype=torch.float32))
    
    if skipped:
        print(f"Skipped {skipped}/{len(folder_paths)} samples due to timeout ({timeout}s)")
    
    return torch.cat(X_list), torch.cat(y_list)



if __name__ == "__main__":
    folder_path = "wrp/data/DungeonMaps/train"
    X, y = generate_N_training_data_for_online_learning_from_folder(folder_path, discounted_step=10, grazing_walls=True, los_type="square360", vision_radius=4, timeout=600)
    print("Generated training data shapes:", X.shape, y.shape)
    file_path = f"wrp/data/dungeon_online_data_16x16_los_square360_vision_4_{X.shape[0]}_samples_train.pt"
    save_data_to_disk(X, y, file_path)