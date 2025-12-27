import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
# Ensure the file path is correct
df = pd.read_csv('Dataset .csv')

# 2. Preprocessing

# Handle missing values
# 'Cuisines' is critical for content-based filtering, so we fill NaNs with an empty string
df['Cuisines'] = df['Cuisines'].fillna('')

# Encode Categorical Variables (as a standard preprocessing step)
# We create new encoded columns for categorical features
le = LabelEncoder()
categorical_cols = ['Has Table booking', 'Has Online delivery', 'Is delivering now']
for col in categorical_cols:
    df[col + '_encoded'] = le.fit_transform(df[col])

# 3. Implement Content-Based Filtering

# We will use 'Cuisines' as the main feature for similarity.
# TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numerical vectors.
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Cuisines'])

def recommend_restaurants(user_preference, df, tfidf_matrix, top_n=5):
    """
    Recommend restaurants based on user preferences using Cosine Similarity.
    
    Args:
    user_preference (dict): Dictionary with keys 'cuisine', 'price_range', 'city'.
    df (pd.DataFrame): The restaurant dataset.
    tfidf_matrix: The pre-computed TF-IDF matrix for cuisines.
    top_n (int): Number of recommendations to return.
    """
    
    # Create a copy to avoid modifying the original dataframe
    filtered_df = df.copy()
    
    # Step 1: Filter by City (if provided)
    if 'city' in user_preference and user_preference['city']:
        filtered_df = filtered_df[filtered_df['City'].str.lower() == user_preference['city'].lower()]
        if filtered_df.empty:
            return f"No restaurants found in {user_preference['city']}"

    # Step 2: Filter by Price Range (if provided)
    # We look for restaurants matching the user's price preference
    if 'price_range' in user_preference:
        filtered_df = filtered_df[filtered_df['Price range'] == user_preference['price_range']]
        # If no exact match, fallback to including cheaper options could be added here
        if filtered_df.empty:
             return "No restaurants found matching price criteria."

    # Step 3: Calculate Similarity
    # Transform user's cuisine preference into a TF-IDF vector
    user_cuisine_vec = tfidf.transform([user_preference['cuisine']])
    
    # Calculate cosine similarity between user preference and ALL restaurants
    # (We compute for all to keep index alignment simple, then subset)
    cosine_sim = cosine_similarity(user_cuisine_vec, tfidf_matrix)
    
    # Add similarity score to the dataframe
    df['similarity'] = cosine_sim[0]
    
    # Filter the results to only include the valid rows (City/Price filtered)
    results = df.loc[filtered_df.index]
    
    # Sort by Similarity (descending) and then by Aggregate Rating (descending)
    results = results.sort_values(by=['similarity', 'Aggregate rating'], ascending=[False, False])
    
    # Return the top N recommendations
    return results[['Restaurant Name', 'Cuisines', 'Price range', 'City', 'Aggregate rating']].head(top_n)

# 4. Test the Recommendation System

# Example 1: User wants Italian food in New Delhi with Price Range 2
user_1 = {
    'cuisine': 'Italian',
    'price_range': 2,
    'city': 'New Delhi'
}

# Example 2: User wants North Indian food in Gurgaon with Price Range 1 (Cheap)
user_2 = {
    'cuisine': 'North Indian',
    'price_range': 1,
    'city': 'Gurgaon'
}

print("--- Recommendations for User 1 (Italian, Price 2, New Delhi) ---")
print(recommend_restaurants(user_1, df, tfidf_matrix))

print("\n--- Recommendations for User 2 (North Indian, Price 1, Gurgaon) ---")
print(recommend_restaurants(user_2, df, tfidf_matrix))