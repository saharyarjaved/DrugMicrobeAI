import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

print("All libraries imported successfully!")

arr = np.array([10, 20, 30])
print("NumPy:", arr)

df = pd.DataFrame({
    "Drug": ["DrugA", "DrugB"],
    "Microbe": ["M1", "M2"]
})

print(df)

G = nx.Graph()
G.add_edge("DrugA", "M1")

print("Nodes:", G.nodes())
print("Edges:", G.edges())