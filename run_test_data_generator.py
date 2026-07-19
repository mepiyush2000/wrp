import argparse

from data_generator import *
from wrp_solver_opt import *
from data_generator import _solve_grid
from utils import *
import os
from tqdm import tqdm

# Generate dta and save only the wrp grid, start and path.


if __name__ == "__main__":
    # Example usage:   
    argparser = argparse.ArgumentParser(description="Generate training data for WRP online learning")
    argparser.add_argument("--num_samples", type=int, required=True, help="Number of training samples to generate")
    argparser.add_argument("--polygon_type", type=str, default="simple", choices=["simple", "holes"], help="Type of polygon to generate")
    argparser.add_argument("--los_type", type=str, default="los4", choices=["los4", "bresenham", "los8", "square360"], help="Type of Line of Sight (LOS) calculation to use for online learning data generation")
    argparser.add_argument("--vision_radius", type=int, default=float('inf'), help="Vision radius for LOS calculations (only applicable for online learning)")

    args = argparser.parse_args()
    num_samples = args.num_samples
    grid_size = (16, 16)
    density = 5
    timeout = 180

    
    file_path = f"data/wrp_gt_data_16x16_los_{args.los_type}_vision_{str(args.vision_radius)}_{num_samples}_samples_{args.polygon_type}.npy"

    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Please delete it or choose a different configuration.")
        exit(1)
    
    skipped = 0
    data_to_save = {"grid": [], "start": [], "path_opt": []}
    for i in tqdm(range(num_samples), desc="Generating samples"):
        gen = WRPDataGenerator(*grid_size)
        if (args.polygon_type == "simple"):
            grid, start = gen.generate_simple_polygon_grid()
        else:
            grid, start = gen.generate_valid_grid(density=density)

        try:
            path_opt, _ = run_with_timeout(_solve_grid, args=(grid, start, args.los_type, args.vision_radius, False), timeout=timeout)  
        except TimeoutError:
            skipped += 1   
            continue

        data_to_save["grid"].append(np.array(grid, dtype=np.uint8))
        data_to_save["start"].append(np.array(start, dtype=np.uint8))
        data_to_save["path_opt"].append(np.array(path_opt, dtype=np.uint8))


    print(f"Generated {num_samples} samples, skipped {skipped} due to timeouts.")
    np.save(file_path, data_to_save)

    print(f"Data saved to {file_path}")



# how to run
# python3 run_test_data_generator.py --polygon_type holes --los_type bresenham --vision_radius 8 --num_samples 100 