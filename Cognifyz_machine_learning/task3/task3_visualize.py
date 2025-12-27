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
# ---------------------------------------------------------
# 7. VISUALIZATION (Add this part to generate images)
# ---------------------------------------------------------
import seaborn as sns

# Plot 1: Feature Importance (What drives the classification?)
# This helps you see which factors (like Cost or City) help predict the Cuisine.
importances = rf_classifier.feature_importances_
feature_names = X.columns
feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')
plt.title('Feature Importance for Cuisine Classification')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('task3_feature_importance.png') # Saves the file
print("\nImage saved: task3_feature_importance.png")
plt.show()

# Plot 2: Confusion Matrix (Where is the model guessing wrong?)
# This visualizes the "Biases". For example, if many 'Chinese' restaurants 
# are predicted as 'North Indian', you will see it here.
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Cuisine')
plt.ylabel('Actual Cuisine')
plt.title('Confusion Matrix: Actual vs Predicted')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('task3_confusion_matrix.png') # Saves the file
print("Image saved: task3_confusion_matrix.png")
plt.show()