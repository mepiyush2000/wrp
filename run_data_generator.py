import argparse
from data_generator import *


if __name__ == "__main__":
    # Example usage:   
    argparser = argparse.ArgumentParser(description="Generate training data for WRP online learning")
    argparser.add_argument("--num_samples", type=int, required=True, help="Number of training samples to generate")

    args = argparser.parse_args()
    num_samples = args.num_samples
    grid_size = (16, 16)
    density = 5
    timeout = 300
    
    X, y = generate_N_training_data_for_online_learning(num_samples, grid_size, density, timeout)
    print(f"Generated {X.shape[0]} training samples with shape {X.shape[1:]} and labels with shape {y.shape[1:]}")
    save_data_to_disk(X, y, f"data/wrp_online_frontier_data_16x16_{num_samples}_samples_SP_train.pt")


# How to run:
# python run_data_generator.py --num_samples 1000