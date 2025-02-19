import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load dataset
def load_data(train_path):
    return pd.read_csv(train_path)

# Preprocess data
def preprocess_data(df):

    features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'SalePrice']
    df = df[features].dropna()

    df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath']

    X = df[['GrLivArea', 'BedroomAbvGr', 'TotalBath']]
    y = df['SalePrice']

    return X, y

# Split dataset
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Main function
def main():
    train_path = 'data/train.csv'
    X, y = preprocess_data(load_data(train_path))
    X_train, X_val, y_train, y_val = split_data(X, y)

    os.makedirs('data', exist_ok=True)

    # Save processed data
    X_train.to_csv('data/X_train.csv', index=False)
    X_val.to_csv('data/X_val.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_val.to_csv('data/y_val.csv', index=False)
    print("Data processing complete. Saved processed files.")

if __name__ == "__main__":
    main()
