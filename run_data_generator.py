import argparse
from data_generator import *
import os


if __name__ == "__main__":
    # Example usage:   
    argparser = argparse.ArgumentParser(description="Generate training data for WRP online learning")
    argparser.add_argument("--num_samples", type=int, required=True, help="Number of training samples to generate")
    argparser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Whether to generate training or test data")
    argparser.add_argument("--type", type=str, default="offline", choices=["offline", "online"], help="Whether to generate data for offline learning or online learning ")
    argparser.add_argument("--discounted_step", type=int, default=0, help="Number of initial steps to discount in the path for training data generation")

    args = argparser.parse_args()
    num_samples = args.num_samples
    split = args.split
    data_type = args.type
    discounted_step = args.discounted_step
    grid_size = (16, 16)
    density = 5
    timeout = 300

    if data_type == "offline":  
        file_path = f"data/wrp_data_16x16_{num_samples}_samples_SP_{split}.pt"
    else:
        file_path = f"data/wrp_online_data_16x16_{num_samples}_samples_SP_{split}.pt"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Please delete it or choose a different configuration.")
        exit(1)


    if data_type == "offline":
        X, y = generate_N_training_data(num_samples, grid_size, density, timeout)
    else:
        X, y = generate_N_training_data_for_online_learning(num_samples, grid_size, density, discounted_step, timeout)

    print(f"Generated {X.shape[0]} training samples with shape {X.shape[1:]} and labels with shape {y.shape[1:]}")
    save_data_to_disk(X, y, file_path)


# How to run:
# python run_data_generator.py --split train --type online --discounted_step 0 --num_samples 251