# main.py

import os

def main():
    print("Starting house price prediction pipeline...")

    # Step 1: Data Preparation
    print("Running data preparation...")
    os.system("python scripts/data_preparation.py")

    # Step 2: Model Training
    print("Training model...")
    os.system("python scripts/model_training.py")

    # Step 3: Making a Prediction
    print("Making a sample prediction...")
    os.system("python scripts/predict.py")

if __name__ == "__main__":
    main()
