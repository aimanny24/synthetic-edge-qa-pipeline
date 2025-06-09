import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import train_model
from src.data_generator import generate_edge_cases
import pandas as pd

# Test that the number of model predictions matches the number of synthetic inputs
def test_model_output_shape():
    model = train_model("data/real_data.csv")
    df_synth = generate_edge_cases()
    preds = model.predict(df_synth)
    assert len(preds) == len(df_synth), "Prediction length does not match input rows"

# Test that model predictions are within the expected class labels (0 or 1 for fraud detection)
def test_model_prediction_values():
    model = train_model("data/real_data.csv")
    df_synth = generate_edge_cases()
    preds = model.predict(df_synth)
    assert set(preds).issubset({0, 1}), "Predictions contain values other than 0 or 1"

# Test that the synthetic data includes all expected columns (Time, V1–V28, Amount)
def test_generated_data_columns():
    df = generate_edge_cases()
    expected_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"

# Test that the schema (column names) of the synthetic data matches the real training data
def test_schema_alignment():
    real = pd.read_csv("data/real_data.csv")
    synth = generate_edge_cases()
    assert list(real.drop("Class", axis=1).columns) == list(synth.columns), "Schema mismatch"
