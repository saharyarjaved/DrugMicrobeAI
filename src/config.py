import torch

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model (Upgraded capacity for 90%+ accuracy)
HIDDEN_DIM = 256
OUTPUT_DIM = 64
HEADS = 4

# Training (Optimized for deeper convergence to 90%+ metrics)
LEARNING_RATE = 0.0003
EPOCHS = 400

# Seed
SEED = 42