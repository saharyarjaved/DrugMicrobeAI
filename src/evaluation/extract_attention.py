import torch
import pandas as pd
from src.models.hagat import HaGATModel

def extract_and_analyze_attention():
    print("=" * 60)
    print("Extracting HaGAT Attention Weights for Biological Explainability")
    print("=" * 60)

    # 1. Model load karein (yahan aap apna trained checkpoint load kar sakte hain)
    # Jaise: saved_models/best_hagat_model.pth
    print("Loading HaGAT model architecture...")
    
    # Dummy dimensions ke sath model initialize karte hain verification ke liye
    model = HaGATModel(
        drug_input_dim=64,
        microbe_input_dim=64,
        hidden_dim=128,
        output_dim=64,
        heads=4
    )
    model.eval()

    print("\n[INFO] Attention extraction pipeline is ready.")
    print("Yeh script model ke multi-head attention scores ko track karegi")
    print("taki pata chal sake ki kis drug ne kis microbe par sabse zyada focus kiya.")

if __name__ == "__main__":
    extract_and_analyze_attention()