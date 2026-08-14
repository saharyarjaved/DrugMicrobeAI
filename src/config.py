import torch

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
HIDDEN_DIM = 128
OUTPUT_DIM = 64
HEADS = 4

# Training
LEARNING_RATE = 0.001
EPOCHS = 100

# Seed
SEED = 42