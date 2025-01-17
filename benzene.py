import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to load and clean the dataset
def load_and_clean_dataset(file_path):
    try:
        print("Loading the dataset...\n")
        df = pd.read_csv(file_path, sep=";")
        print("Dataset loaded successfully!\n")
        
        # Replace commas with periods in numeric columns
        for col in df.columns:
            if df[col].dtype == 'object':  # Check for object type (likely strings)
                df[col] = df[col].str.replace(',', '.', regex=False)
        
        # Convert all columns to appropriate numeric types where possible
        df = df.apply(pd.to_numeric, errors='coerce')
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        exit()
    except Exception as e:
        print(f"Error loading the dataset: {e}")
        exit()

# Load the dataset
file_path = "AirQualityUCI.csv"  # Path to your dataset
dataset = load_and_clean_dataset(file_path)

# Inspect column names to ensure correct feature selection
print("\nColumn names in the dataset:")
print(dataset.columns)

# Drop rows with missing values in the selected feature columns
feature_columns = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'NOx(GT)', 'T', 'RH', 'AH']  # Features
target_column = 'C6H6(GT)'  # Air quality metric (Benzene concentration)
dataset = dataset.dropna(subset=feature_columns + [target_column])

# Prepare feature and target datasets
X = dataset[feature_columns]  # Features (independent variables)
y = dataset[target_column]   # Target variable (dependent variable, air quality)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model's performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# Display evaluation metrics
print("\nModel Evaluation Metrics:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")

# Visualize predictions vs actual values for test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label="Ideal Prediction")
plt.title("Actual vs Predicted Benzene Concentration (C6H6(GT))")
plt.xlabel("Actual Benzene Concentration")
plt.ylabel("Predicted Benzene Concentration")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Analyze feature importance
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df)

# Plot feature importance
plt.figure(figsize=(8, 5))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.title("Feature Importance in Predicting Air Quality (C6H6)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()