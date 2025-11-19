import numpy as np
import torch

# Define the random seed for reproducibility
RANDOM_SEED = 100
np.random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Specify the device CPU or GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Specify the S & P data path
S_P_DATA_PATH = './data/sp500_stocks.csv'  

# Data partition fractions
TEST_SIZE = 0.15
VAL_SIZE = 0.1765 # 0.1765 of the remaining 0.85 gives ~0.15 of total

# Simple Moving Average (SMA) windows
SMA_SHORT_WINDOW = 20
SMA_LONG_WINDOW = 50

# Console output settings
LINE_DIVIDER = "-" * 50
CHARRIAGE_RETURN = "\n"

# Deep neural network configurations
DNN_INPUT_SIZE = 10  # Example input size, adjust as needed 
DNN_HIDDEN_SIZE1 = 64
DNN_HIDDEN_SIZE2 = 32   
DNN_OUTPUT_SIZE = 1  # Binary classification
DNN_DROPOUT_RATE = 0.3
DNN_LEARNING_RATE = 0.001
DNN_NUM_EPOCHS = 50
DNN_BATCH_SIZE = 32
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_VERBOSE = True
EARLY_STOPPING_TOLERANCE = 1e-6
EARLY_STOPPING_PATIENCE = 5
