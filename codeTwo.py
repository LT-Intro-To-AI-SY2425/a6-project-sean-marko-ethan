import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to load the dataset and handle potential errors
def load_dataset(file_path):
    try:
        print("Loading the dataset...\n")
        df = pd.read_csv(file_path, sep=";")  # Read the CSV file into a pandas DataFrame
        print("Dataset loaded successfully!\n")
        print("First few rows of the dataset:")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        exit()
    except Exception as e:
        print(f"Error loading the dataset: {e}")
        exit()

# Specify the correct file path to your dataset
file_path = "AirQualityUCI.csv"  # Ensure this path is correct

# Load the dataset
dataset = load_dataset(file_path)

# Inspect column names to ensure correct feature selection
print("\nColumn names in the dataset:")
print(dataset.columns)

# Convert Date column to datetime if it's not already
dataset['Date'] = pd.to_datetime(dataset['Date'], errors='coerce')

# Replace commas in the numeric columns (if any), and convert all columns to numeric
dataset = dataset.replace({',': '.'}, regex=True)  # Replace commas with dots
dataset = dataset.apply(pd.to_numeric, errors='coerce')  # Convert all values to numeric, invalid ones become NaN

# Drop rows with missing values in the selected feature columns only
feature_columns = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'NOx(GT)', 'C6H6(GT)', 'PT08.S3(NOx)', 'T', 'RH', 'AH']
dataset = dataset.dropna(subset=feature_columns)

# Ensure all feature columns exist in the dataset
missing_columns = [col for col in feature_columns if col not in dataset.columns]
if missing_columns:
    print(f"Error: The following columns are missing from the dataset: {missing_columns}")
    exit()

X = dataset[feature_columns]  # Features (independent variables)
y = dataset['PT08.S1(CO)']   # Target variable (dependent variable)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for many models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make a prediction for the entire dataset
X_scaled = scaler.transform(X)  # Standardize the entire feature set
predictions = model.predict(X_scaled)

# Compute the average prediction and actual CO level
average_prediction = predictions.mean()
average_actual = y.mean()

# Print the predictions and the actual CO levels for all the samples
print("\nPredictions vs Actual CO levels for all samples:")
for i in range(len(dataset)):  # Iterate through all the samples
    print(f"Sample {i+1} - Predicted CO level: {predictions[i]:.2f}, Actual CO level: {y.iloc[i]:.2f}")

# Print the average prediction and actual CO level
print(f"\nAverage predicted CO level for all samples: {average_prediction:.2f}")
print(f"Average actual CO level for all samples: {average_actual:.2f}")

# Evaluate the model's performance on the test set
y_pred = model.predict(X_test_scaled)
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