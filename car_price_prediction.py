import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Sample Data Creation (You should replace this with your actual dataset)
data = {
    'model': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    'horse_power': [100, 150, 200, 250, 300, 350, 400, 450],
    'distance_travelled': [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000],
    'year_of_manufacture': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    'fuel_type': ['Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel'],
    'mileage': [15, 18, 20, 22, 25, 30, 35, 40],
    'color': ['Red', 'Blue', 'Green', 'Black', 'White', 'Yellow', 'Silver', 'Gray'],
    'safety_feature': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No'],
    'price': [20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Preprocessing
# Convert categorical variables to numerical
label_encoders = {}
for column in ['model', 'fuel_type', 'color', 'safety_feature']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Features and target variable
X = df.drop('price', axis=1)
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

# Evaluation Metrics
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

# Random Forest Evaluation
rf_mse, rf_r2 = evaluate_model(y_test, rf_predictions)
print(f"Random Forest - MSE: {rf_mse}, R²: {rf_r2}")

# XGBoost Evaluation
xgb_mse, xgb_r2 = evaluate_model(y_test, xgb_predictions)
print(f"XGBoost - MSE: {xgb_mse}, R²: {xgb_r2}")

# Visualizations
plt.figure(figsize=(15, 10))

# Scatter plot for Horse Power vs Price
plt.subplot(2, 2, 1)
sns.scatterplot(data=df, x='horse_power', y='price', hue='fuel_type', style='safety_feature', s=100)
plt.title('Horse Power vs Price')
plt.xlabel('Horse Power')
plt.ylabel('Price')

# Scatter plot for Distance Travelled vs Price
plt.subplot(2, 2, 2)
sns.scatterplot(data=df, x='distance_travelled', y='price', hue='fuel_type', style='safety_feature', s=100)
plt.title('Distance Travelled vs Price')
plt.xlabel('Distance Travelled')
plt.ylabel('Price')

# Box plot for Price by Fuel Type
plt.subplot(2, 2, 3)
sns.boxplot(data=df, x='fuel_type', y='price')
plt.title('Price Distribution by Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Price')

# Pair plot for all features against Price
plt.subplot(2, 2, 4)
sns.pairplot(df, x_vars=['horse_power', 'distance_travelled', 'year_of_manufacture', 'mileage'], y_vars='price', height=4, aspect=1, kind='scatter')
plt.suptitle('Pair Plot of Features vs Price', y=1.02)

plt.tight_layout()
plt.show()
