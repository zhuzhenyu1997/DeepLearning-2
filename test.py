import os
import argparse
import torch
from solver import Solver

if __name__ == '__main__':
    # Set configs and hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch_num', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--log_path', type=str, default='./log')
    parser.add_argument('--log_file', type=str, default='logs.txt')
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=25)
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--normalize', type=bool, default=True)
    config = parser.parse_args()
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)

    # Turn off the auto_grad
    torch.set_grad_enabled(False)

    # Training
    solver = Solver(config)
    solver.train()