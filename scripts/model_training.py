import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os


# Load processed data
def load_processed_data():
    X_train = pd.read_csv('data/X_train.csv')
    X_val = pd.read_csv('data/X_val.csv')
    y_train = pd.read_csv('data/y_train.csv')
    y_val = pd.read_csv('data/y_val.csv')
    return X_train, X_val, y_train, y_val


# Train model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# Evaluate model
def evaluate_model(model, X_val, y_val):
    predictions = model.predict(X_val)
    mse = mean_squared_error(y_val, predictions)
    rmse = mse ** 0.5
    return rmse


# Main function
def main():
    X_train, X_val, y_train, y_val = load_processed_data()
    model = train_model(X_train, y_train)
    rmse = evaluate_model(model, X_val, y_val)

    print(f'Validation RMSE: {rmse}')

    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)

    # Save model
    joblib.dump(model, 'models/linear_regression_model.pkl')
    print("Model training complete. Saved model to models/linear_regression_model.pkl")


if __name__ == "__main__":
    main()
