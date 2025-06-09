# run_pipeline.py
# Manual integration test for full ML QA pipeline

import os
import pandas as pd
from src.model import train_model
from src.data_generator import generate_edge_cases
from src.validation import validate_data

def main():
    try:
        # Step 0: Ensure data folder exists
        os.makedirs("data", exist_ok=True)

        # Step 1: Train the model
        print("Training model...")
        model = train_model("data/real_data.csv")

        # Step 2: Generate synthetic data
        print("Generating synthetic test data...")
        df_synth = generate_edge_cases()
        print(f"Synthetic data shape: {df_synth.shape}")

        # Step 3: Save synthetic data
        df_synth.to_csv("data/synthetic_data.csv", index=False)
        print("Saved synthetic_data.csv")

        # Step 4: Validate synthetic data
        print("Validating synthetic data...")
        validate_data(df_synth)

        # Step 5: Predict on synthetic data
        print("Running predictions...")
        predictions = model.predict(df_synth)

        # Step 6: Show and save predictions
        print("First 10 predictions:", predictions[:10])
        df_synth["prediction"] = predictions
        df_synth.to_csv("data/synthetic_data_with_predictions.csv", index=False)
        print("Saved synthetic_data_with_predictions.csv")

    except Exception as e:
        print("ERROR during pipeline execution:", e)

if __name__ == "__main__":
    main()
