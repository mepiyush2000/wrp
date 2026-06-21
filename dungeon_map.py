import json

import numpy as np
from tqdm import tqdm
import os
import torch
from wrp_solver_suboptimal import *
from data_generator import generate_training_data_for_online_learning
from utils import *

def _solve_grid_subopt(grid, start, los_type = "los4", vision_radius = float('inf')):
    solver = WRPSolverJF(grid, start, los_type=los_type, vision_radius=vision_radius)
    return solve_wrp_jf(solver, weight=1, df=6, heuristic="tsp")

def generate_N_training_data_for_online_learning_from_folder(folder_path, discounted_step, grazing_walls, los_type, vision_radius, timeout=600):
    X_list = []
    y_list = []
    skipped = 0
    
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    folder_paths = sorted(png_files)[4000:]

    for imname in tqdm(folder_paths):
        if (imname[-4:]!=".png"): continue
        impath = os.path.join(folder_path, imname)
        # Generate a random grid and path
        grid, start = image_to_grid2(impath)
        resJsonPath = impath.replace(".png", "_vision_3.npy")
        if os.path.exists(resJsonPath): continue

        try:
            path_opt, _ = run_with_timeout(_solve_grid_subopt, args=(grid, start, los_type, vision_radius), timeout=timeout)
            resJson = {"grid": grid, "start": start, "path_opt": path_opt}
            np.save(resJsonPath, resJson)

        except TimeoutError:
            skipped += 1
            continue
        
        # # Generate training data from the path
        # X, y = generate_training_data_for_online_learning(grid, path_opt, discounted_step=discounted_step, grazing_walls=grazing_walls, los_type=los_type, vision_radius=vision_radius)
        # X_list.append(torch.tensor(X, dtype=torch.float32))
        # y_list.append(torch.tensor(y, dtype=torch.float32))
    
    if skipped:
        print(f"Skipped {skipped}/{len(folder_paths)} samples due to timeout ({timeout}s)")
    
    return torch.cat(X_list), torch.cat(y_list)



if __name__ == "__main__":
    folder_path = "data/DungeonMaps/train"
    X, y = generate_N_training_data_for_online_learning_from_folder(folder_path, discounted_step=10, grazing_walls=True, los_type="square360", vision_radius=3, timeout=600)
    print("Generated training data shapes:", X.shape, y.shape)
    file_path = f"data/dungeon_online_data_16x16_los_square360_vision_4_{X.shape[0]}_samples_test.pt"
    save_data_to_disk(X, y, file_path)