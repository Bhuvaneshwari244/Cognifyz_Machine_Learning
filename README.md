# Cognifyz_Machine_Learning

🍽️ Restaurant Data Analysis & Prediction - Machine Learning Internship
This repository contains the source code and analysis for the Machine Learning Internship tasks completed at Cognifyz Technologies. The project focuses on analyzing restaurant data to predict ratings, recommend restaurants, classify cuisines, and visualize geospatial patterns.

📂 Dataset
File: Dataset .csv

Description: A dataset containing restaurant details including location, cuisines, average cost, ratings, and votes.

🛠️ Technologies Used
Language: Python 🐍

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

🚀 Task Breakdown
## 📊 Task 1: Predict Restaurant Ratings
Objective: Build a machine learning model to predict the Aggregate Rating of a restaurant based on other features.

Approach:

Preprocessed data (handled missing values, encoded categorical features).

Trained Linear Regression and Decision Tree Regressor models.

Key Results:

Best Model: Decision Tree Regressor.

Accuracy: Achieved an R-squared score of ~92%.

Insight: Feature Importance analysis revealed that the "Number of Votes" is the most critical factor affecting a restaurant's rating.

## Output (Visualization)
https://www.linkedin.com/posts/bhuvaneshwari-rebba-650800280_machinelearning-datascience-predictiveanalytics-activity-7410719486594650112-Sx5u?utm_source=share&utm_medium=member_desktop&rcm=ACoAAER2Y5wBFQn-3KxbyzrMce3aWEi7zP3K_os

## 🍕 Task 2: Restaurant Recommendation System
Objective: Create a content-based recommendation system based on user preferences.

Approach:

Used TF-IDF Vectorization on the 'Cuisines' column.

Calculated Cosine Similarity to find matches.

Implemented filters for City and Price Range.

Key Results:

The system successfully recommends top-rated restaurants with high similarity scores based on user input (e.g., "Italian" in "New Delhi").

## Output (Visualization)
https://www.linkedin.com/posts/bhuvaneshwari-rebba-650800280_recommendationsystem-nlp-ai-activity-7410721444587876352-OAgX?utm_source=share&utm_medium=member_desktop&rcm=ACoAAER2Y5wBFQn-3KxbyzrMce3aWEi7zP3K_os

## 🥡 Task 3: Cuisine Classification
Objective: Develop a model to classify restaurants based on their cuisines.

Approach:

Target variable: Main Cuisine (extracted from the list).

Algorithm: Random Forest Classifier.

analyzed model performance using a Confusion Matrix.

Key Results:

Accuracy: ~40% (Due to highly imbalanced data).

Insight: The dataset is heavily biased towards "North Indian" cuisine, causing the model to favor the majority class. This highlights the real-world challenge of class imbalance.

## Output (Visualization)
https://www.linkedin.com/posts/bhuvaneshwari-rebba-650800280_machinelearning-classification-randomforest-activity-7410723607238062080-_lGX?utm_source=share&utm_medium=member_desktop&rcm=ACoAAER2Y5wBFQn-3KxbyzrMce3aWEi7zP3K_os

## 🌍 Task 4: Location-based Analysis
Objective: Perform a geographical analysis of the restaurants.

Approach:

Visualized distribution using Latitude/Longitude scatter plots.

Analyzed restaurant concentration by city.

Compared average ratings across different cities.

Key Results:

Highest Concentration: New Delhi has the highest number of restaurants.

Highest Quality: Cities like London and Orlando have the highest average ratings, despite having fewer restaurants than Indian cities.

## Output (Visualization)
https://www.linkedin.com/posts/bhuvaneshwari-rebba-650800280_datavisualization-geospatialanalysis-python-activity-7410725660991774720-o4XT?utm_source=share&utm_medium=member_desktop&rcm=ACoAAER2Y5wBFQn-3KxbyzrMce3aWEi7zP3K_os

## ⚙️ How to Run the Code
Clone the repository:

git clone https://github.com/yourusername/cognifyz-ml-internship.git

Install dependencies:

pip install pandas numpy scikit-learn matplotlib seaborn

Run the scripts:

python Task1.py  # For Rating Prediction
python Task2.py  # For Recommendation System
python Task3.py  # For Cuisine Classification
python Task4.py  # For Location Analysis

📢 Author
Bhuvaneshwari Rebba

Role: Machine Learning Intern at Cognifyz Technologies

LinkedIn: www.linkedin.com/in/bhuvaneshwari-rebba-650800280

Email: bhuvaneshwaritsms010@gmail.com

This project was completed as part of the Cognifyz Technologies Internship Program.
