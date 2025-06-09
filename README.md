# Project Name: 
Synthetic Data QA Pipeline for ML Model Testing

# Project Summary
This project demonstrates how to build a QA automation pipeline to test machine learning classification models using synthetic data to simulate rare and boundary case inputs. The project specifically focuses on a Credit Card Fraud Detection model and ensures the model's robustness, fairness, and reliability using tools like Great Expectations and PyTest.

# Project Details
  • Project Overview: Create a QA automation pipeline that tests an ML classification model (e.g., fraud detection) using synthetic data to simulate rare and boundary cases. The goal is to validate the model's robustness and identify weaknesses in edge conditions.
  
  • Goal: To simulate real-world AI QA tasks by building a test automation pipeline that trains a real ML model, generates synthetic edge-case data, validates this data against business rules, and runs automated tests to ensure model stability, reliability, and fairness.

# Project Tech Stack
    • Python 3.9+
    • scikit-learn – ML model training
    • Faker or SDV – synthetic data generation
    • Great Expectations – data quality and schema validation
    • PyTest – test automation
    • pandas, numpy, matplotlib – data handling and visualization

# Step-by-Step Build Plan
    Step 1: Set Up Your Environment
        •	Install Python 3.9+ from https://www.python.org/downloads/
        •	Create a virtual environment: `python -m venv venv`
        •	Activate the environment: `venv\Scripts\activate`
        •	Install dependencies using: `pip install -r requirements.txt`
        •	Output: requirements.txt, ready-to-run project folder in VS Code

    Step 2: Train a Base ML Model (Fraud Detection)
        •	Use the Kaggle dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
        •	Rename `creditcard.csv` to `real_data.csv` and place in `data/`
        •	Update `src/model.py` to train a RandomForestClassifier on the fraud detection data
        •	Run: `python run_pipeline.py` to train and print the classification report

    Step 3: Generate Synthetic Edge Case Data
        •	Create `src/data_generator.py` with a `generate_edge_cases()` function
        •	Use Faker to generate edge values for Amount, Time, and synthetic transaction data
        •	Save to `data/synthetic_data.csv`

    Step 4: Validate the Synthetic Data (Great Expectations)
        •	Create `src/validation.py` to validate columns using Great Expectations
        •	Validate numerical range, column presence, and basic schema
        •	Print or log the validation results to confirm expectations are met

    Step 5: Run and Test the Pipeline
        •	The `python run_pipeline.py` script ties the entire QA workflow together. It performs the following actions:
            1. Trains the fraud detection ML model** using `real_data.csv`
            2. Generates synthetic edge-case data** that simulates rare or boundary conditions
            3. Validates the synthetic data** against business rules using Great Expectations
            4. Runs model inference** on synthetic data to test prediction behavior
            5. Prints sample predictions** to verify that the model can handle edge inputs gracefully
    
        •	This script ensures that all major ML testing steps (training, data simulation, validation, and testing) are executed in one end-to-end flow.
        •	What this script (run_pipeline.py) achieves:
            1.	Automate the full ML testing pipeline
            2.	Ensures your model can process non-standard, synthetic inputs
            3.	Validate data schema and values before testing
            4.	Gives quick feedback on how the model responds to rare conditions

        


