# 1. The synthetic data has the same schema as the training data
# 2. The model will accept it without errors


import pandas as pd
import numpy as np

def generate_edge_cases(n=100):
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    data = []

    for _ in range(n):
        row = {
            "Time": np.random.uniform(0, 172792),
            "Amount": np.random.uniform(10000, 50000)
        }
        for i in range(1, 29):
            row[f"V{i}"] = np.random.normal(0, 1)
        data.append(row)

    return pd.DataFrame(data)
