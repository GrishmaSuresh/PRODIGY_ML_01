import pandas as pd
import joblib
import numpy as np

# Load model
def load_model(model_path):
    return joblib.load(model_path)

# Predict house price
def predict_house_price(model, square_feet, bedrooms, full_bath, half_bath):
    total_bath = full_bath + (0.5 * half_bath)
    features = pd.DataFrame([[square_feet, bedrooms, total_bath]],
                            columns=['GrLivArea', 'BedroomAbvGr', 'TotalBath'])
    prediction = model.predict(features)
    return prediction.item()

# Main function
def main():
    model_path = 'models/linear_regression_model.pkl'
    model = load_model(model_path)

    # Example prediction
    square_feet = int(input("Enter the total square feet of house: "))
    bedrooms = int(input("Enter the number of bedrooms: "))
    full_bath = int(input("Enter the number of full bathrooms: "))
    half_bath = int(input("Enter the number of half bathrooms: "))

    predicted_price = predict_house_price(model, square_feet, bedrooms, full_bath, half_bath)
    print(f'Predicted House Price: ${predicted_price:.2f}')

if __name__ == "__main__":
    main()
