import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Data
# Adjust path if necessary
df = pd.read_csv('Dataset .csv')

# 2. Preprocessing

# Handle missing values in 'Cuisines'
df = df.dropna(subset=['Cuisines'])

# The 'Cuisines' column often contains multiple items (e.g., "Italian, Pizza").
# We simplify the task by predicting the FIRST cuisine listed (the primary cuisine).
df['Main_Cuisine'] = df['Cuisines'].apply(lambda x: x.split(',')[0].strip())

# Filter for the Top 10 Most Common Cuisines
# Classification works best when there are enough examples for each class.
top_cuisines = df['Main_Cuisine'].value_counts().head(10).index
df_filtered = df[df['Main_Cuisine'].isin(top_cuisines)].copy()

print(f"Number of classes after filtering: {df_filtered['Main_Cuisine'].nunique()}")
print(f"Classes: {df_filtered['Main_Cuisine'].unique()}")

# Encode Categorical Features
le = LabelEncoder()
categorical_cols = ['Has Table booking', 'Has Online delivery', 'Is delivering now', 'City']

for col in categorical_cols:
    df_filtered[col] = le.fit_transform(df_filtered[col])

# Encode the Target Variable (Main_Cuisine)
# We store the encoder to map numbers back to names later
target_le = LabelEncoder()
df_filtered['Main_Cuisine_Code'] = target_le.fit_transform(df_filtered['Main_Cuisine'])

# Select Features (X) and Target (y)
feature_cols = ['Average Cost for two', 'Has Table booking', 'Has Online delivery', 'Price range', 'Aggregate rating', 'Votes', 'City']
X = df_filtered[feature_cols]
y = df_filtered['Main_Cuisine_Code']

# 3. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
# We use Random Forest, which is robust for multi-class classification
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# 5. Evaluate Model
y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Detailed Classification Report
target_names = target_le.inverse_transform(sorted(df_filtered['Main_Cuisine_Code'].unique()))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 6. Analyze Performance
# Compare actual vs predicted to see where the model struggles
print("\nConfusion Matrix Analysis (Sample):")
results = pd.DataFrame({'Actual': target_le.inverse_transform(y_test), 'Predicted': target_le.inverse_transform(y_pred)})
print(results.head(10))