import argparse
import numpy as np
from scipy.linalg import pinv, svd # Your only additional allowed imports!

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Assignment 3",
        epilog = "CSCI 4360/6360 Data Science II: Fall 2017",
        add_help = "How to use",
        prog = "python assignment3.py <arguments>")
    parser.add_argument("-f", "--infile", required = True,
        help = "Dynamic texture file, a NumPy array.")
    parser.add_argument("-q", "--dimensions", required = True, type = int,
        help = "Number of state-space dimensions to use.")
    parser.add_argument("-o", "--output", required = True,
        help = "Path where the 1-step prediction will be saved as a NumPy array.")

    args = vars(parser.parse_args())

    # Collect the arguments.
    input_file = args['infile']
    q = args['dimensions']
    output_file = args['output']

    # Read in the dynamic texture data.
    M = np.load(input_file)
    
    ### FINISH ME
