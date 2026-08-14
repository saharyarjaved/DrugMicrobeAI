import numpy as np
import pandas as pd
import matplotlib
import networkx as nx
import sklearn
import torch

print("=" * 50)
print("Environment Check")
print("=" * 50)

print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", matplotlib.__version__)
print("NetworkX:", nx.__version__)
print("Scikit-Learn:", sklearn.__version__)
print("PyTorch:", torch.__version__)

try:
    import torch_geometric
    print("PyTorch Geometric:", torch_geometric.__version__)
except Exception as e:
    print("PyTorch Geometric Error:", e)

print("=" * 50)
print("Everything looks good!")