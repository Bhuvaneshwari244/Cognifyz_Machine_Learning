# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
# Make sure the 'Dataset .csv' file is in the same directory as this script
df = pd.read_csv('Dataset .csv')

# 2. Data Preprocessing

# Drop columns that are not useful for prediction or would cause data leakage
# 'Restaurant ID', 'Restaurant Name' are identifiers.
# 'Address', 'Locality', 'Locality Verbose' are location details (we can use 'City').
# 'Switch to order menu' has only one value 'No'.
# 'Rating color', 'Rating text' are derived from the target 'Aggregate rating' (leakage).
df = df.drop(['Restaurant ID', 'Restaurant Name', 'Address', 'Locality', 
              'Locality Verbose', 'Switch to order menu', 'Rating color', 'Rating text'], axis=1)

# Handle missing values
# 'Cuisines' has some missing values. We will drop those rows.
print(f"Missing values before handling:\n{df.isnull().sum()}")
df = df.dropna(subset=['Cuisines'])

# Encode Categorical Variables
# We use LabelEncoder for categorical columns to convert text to numbers
le = LabelEncoder()
categorical_cols = ['City', 'Cuisines', 'Currency', 'Has Table booking', 
                    'Has Online delivery', 'Is delivering now']

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 3. Split the data into features (X) and target (y)
X = df.drop('Aggregate rating', axis=1)
y = df['Aggregate rating']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Selection and Training

# Initialize models
lin_reg = LinearRegression()
dt_reg = DecisionTreeRegressor(random_state=42)

# Train Linear Regression
lin_reg.fit(X_train, y_train)

# Train Decision Tree Regression
dt_reg.fit(X_train, y_train)

# 5. Model Evaluation
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"--- {model_name} Performance ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")
    return y_pred

print("Model Evaluation Results:")
y_pred_lin = evaluate_model(lin_reg, X_test, y_test, "Linear Regression")
y_pred_dt = evaluate_model(dt_reg, X_test, y_test, "Decision Tree Regressor")

# 6. Analyze Feature Importance (using Decision Tree)
importances = dt_reg.feature_importances_
feature_names = X.columns

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance (Decision Tree)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# Print most influential features
print("\nMost Influential Features:")
print(feature_importance_df.head())