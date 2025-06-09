import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.model import train_model
from src.data_generator import generate_edge_cases

# Test that the model produces only valid binary predictions (0 or 1) when run on synthetic edge-case data
# Expected results: 
# Pass if Every prediction returned by the model is either 0 or 1 (i.e., binary classification).
# Fail if Any value in preds is not 0 or 1, e.g., 2, -1, "fraud", or a float like 0.7.

def test_model_on_synthetic():
    model = train_model("data/real_data.csv")
    synthetic_df = generate_edge_cases(50)
    # preds = model.predict(synthetic_df.drop("fraud", axis=1)) # No 'fraud' column in synthetic data
    preds = model.predict(synthetic_df)
    assert all(pred in [0, 1] for pred in preds)

