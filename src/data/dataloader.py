import pandas as pd
import torch

from src.config import PROCESSED_DATA


class DrugMicrobeDataset:

    def __init__(self):

        self.train = pd.read_csv(PROCESSED_DATA / "train_encoded.csv")
        self.test = pd.read_csv(PROCESSED_DATA / "test_encoded.csv")

    def get_train(self):

        return self.train

    def get_test(self):

        return self.test

    def get_edge_index(self):

        edges = torch.tensor([
            self.train["Drug_ID"].values,
            self.train["Microbe_ID"].values
        ], dtype=torch.long)

        return edges