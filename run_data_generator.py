import argparse
from data_generator import *
import os


if __name__ == "__main__":
    # Example usage:   
    argparser = argparse.ArgumentParser(description="Generate training data for WRP online learning")
    argparser.add_argument("--num_samples", type=int, required=True, help="Number of training samples to generate")
    argparser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Whether to generate training or test data")
    argparser.add_argument("--type", type=str, default="online", choices=["offline", "online"], help="Whether to generate data for offline learning or online learning ")
    argparser.add_argument("--discounted_step", type=int, default=0, help="Number of initial steps to discount in the path for training data generation")
    argparser.add_argument("--grazing", action="store_true", help="Whether to use grazing for data generation (only applicable for online learning)")
    argparser.add_argument("--los_type", type=str, default="los4", choices=["los4", "bresenham", "los8", "square360"], help="Type of Line of Sight (LOS) calculation to use for online learning data generation")
    argparser.add_argument("--vision_radius", type=int, default=np.inf, help="Vision radius for LOS calculations (only applicable for online learning)")

    args = argparser.parse_args()
    num_samples = args.num_samples
    split = args.split
    data_type = args.type
    discounted_step = args.discounted_step # 10
    grazing = args.grazing
    grid_size = (16, 16)
    density = 5
    timeout = 180

    print("Args info:")
    print(f"Number of samples: {num_samples}")
    print(f"Data split: {split}")
    print(f"Data type: {data_type}")
    print(f"Discounted steps: {discounted_step}")
    print(f"Grazing: {grazing}")
    print(f"LOS type: {args.los_type}")
    print(f"Vision radius: {args.vision_radius}")
    print(f"Grid size: {grid_size}")
    print(f"Timeout: {timeout}")

    if data_type == "offline":  
        file_path = f"data/wrp_data_16x16_{num_samples}_samples_SP_{split}.pt"
    else:
        file_path = f"data/wrp_online_data_16x16_los_{args.los_type}_vision_{str(args.vision_radius)}_{num_samples}_samples_SP_{split}.pt"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Please delete it or choose a different configuration.")
        exit(1)


    if data_type == "offline":
        X, y = generate_N_training_data(num_samples, grid_size, density, timeout)
    else:
        X, y = generate_N_training_data_for_online_learning(num_samples, grid_size, density, discounted_step, grazing_walls=grazing, los_type=args.los_type, vision_radius=args.vision_radius, timeout=timeout)

    print(f"Generated {X.shape[0]} training samples with shape {X.shape[1:]} and labels with shape {y.shape[1:]}")
    save_data_to_disk(X, y, file_path)


# How to run:
# python run_data_generator.py --split train --type online --grazing --discounted_step 10 --num_samples 251
#with grazing
# python run_data_generator.py --split train --type online --grazing --discounted_step 10  --los_type bresenham --vision_radius 8 --num_samples 251
# without grazing
# python3 run_data_generator.py --split train --type online --discounted_step 10  --los_type bresenham --vision_radius 8 --num_samples 251