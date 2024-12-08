import pandas as pd
import numpy as np
import os
import joblib  # Import for saving/loading models
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

url = 'https://raw.githubusercontent.com/SilvioManfredDante/434_project/main/IceHockey2023VsSalaryVsPlayer.csv'

print('Welcome To The Model')

def load_and_preprocess_data(url):
    # Load the dataset
    df = pd.read_csv(url)
    
    # Strip spaces from column names
    df.columns = df.columns.str.strip()
    
    # Handle missing and infinite values
    df.fillna(0, inplace=True)
    df.replace([-np.inf, np.inf], 0, inplace=True)
    
    # Clean and preprocess salary columns
    df['Salary'] = df['Salary'].map(lambda x: x.replace(',', '')).astype(int)
    df['Cap Hit'] = df['Cap Hit'].map(lambda x: x.replace(',', '')).astype(int)
    
    # Drop unnecessary columns (assuming 'Year' and other non-essential columns)
    df = df.iloc[:, 0:-6]
    
    # Encode categorical columns like 'Team'
    df['Team'] = LabelEncoder().fit_transform(df['Team'])
    
    # Drop the 'Year' column if exists
    if 'Year' in df.columns:
        df.drop('Year', axis=1, inplace=True)
    
    # Scale numerical columns
    for col in df.columns[2:-2]:
        df[col] = StandardScaler().fit_transform(df[[col]])
    
    return df

def train_model(X_train, y_train, model_path='model.pkl'):
    # Train a RandomForestRegressor model
    model = RandomForestRegressor()
    model.fit(X_train.select_dtypes(include=[float, int]), y_train)
    
    # Save the trained model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model

def load_model(model_path='model.pkl'):
    # Load the model if it exists
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        return joblib.load(model_path)
    else:
        print("No pre-saved model found. Training a new one.")
        return None

def predict_salary(model, df, first_name, last_name):
    # Ensure the columns for first and last name exist
    if 'First Name' not in df.columns or 'Last Name' not in df.columns:
        print("Error: 'First Name' or 'Last Name' columns are missing in the dataset.")
        return None
    
    # Find the player by first and last name
    player_row = df[(df['First Name'] == first_name) & (df['Last Name'] == last_name)]
    
    if player_row.empty:
        print(f"Player {first_name} {last_name} not found.")
        return None
    
    # Prepare the player's features for prediction
    player_features = player_row.drop(['Salary'], axis=1).select_dtypes(include=[float, int])
    prediction = model.predict(player_features)
    
    return prediction[0]

def main():
    # Dataset URL
    url = 'https://raw.githubusercontent.com/SilvioManfredDante/434_project/main/IceHockey2023VsSalaryVsPlayer.csv'
    model_path = 'model.pkl'  # Path to save/load the model
    
    # Load and preprocess data
    df = load_and_preprocess_data(url)
    
    # Split data into training and testing sets
    X = df.drop('Salary', axis=1)
    y = df['Salary']
    X_train = X[:300]
    y_train = y[:300]
    X_test = X[300:]
    y_test = y[300:]
    
    # Try to load the model
    model = load_model(model_path)
    if model is None:
        # Train and save the model if not found
        model = train_model(X_train, y_train, model_path)
    
    # Get prediction for a specific player
    first_name = input("Enter player's first name: ")
    last_name = input("Enter player's last name: ")
    
    prediction = predict_salary(model, df, first_name, last_name)
    if prediction:
        print(f"Predicted Salary for {first_name} {last_name}: ${prediction:,.2f}")
        return prediction

# Run the main function
if __name__ == "__main__":
    main()
