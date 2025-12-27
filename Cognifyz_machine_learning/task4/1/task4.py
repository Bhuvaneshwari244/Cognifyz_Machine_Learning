import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
try:
    df = pd.read_csv('Dataset .csv')
except FileNotFoundError:
    # Use full path if the file is not found in the current directory
    # Update this path to match your specific file location
    df = pd.read_csv(r'C:\Users\vijay kumar\OneDrive\Desktop\Cognifyz_machine_learning\Aggregate rating\Dataset .csv')

# 2. Visualize Distribution on a Map
# We use a scatter plot of Longitude vs Latitude to simulate a map.
# The 'hue' parameter colors points by Country Code to show different regions.
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Longitude', y='Latitude', hue='Country Code', palette='viridis', data=df, alpha=0.6, legend=False)
plt.title('Geographical Distribution of Restaurants')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.savefig('task4_geo_distribution.png')
print("Map visualization saved as 'task4_geo_distribution.png'")
plt.show()

# 3. Analyze Concentration of Restaurants by City
city_counts = df['City'].value_counts()
top_cities = city_counts.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_cities.index, y=top_cities.values, palette='magma')
plt.title('Top 10 Cities with Highest Concentration of Restaurants')
plt.xlabel('City')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=45)
plt.tight_layout() # Adjusts layout to prevent label clipping
plt.savefig('task4_city_concentration.png')
print("City concentration chart saved as 'task4_city_concentration.png'")
plt.show()

# 4. Statistics by City (Avg Rating, Avg Cost)
# Group the data by City and calculate the mean for numeric columns
city_stats = df.groupby('City')[['Aggregate rating', 'Average Cost for two', 'Price range']].mean()

# Filter out cities with very few restaurants (e.g., < 10) to ensure stats are reliable
reliable_cities = city_stats[df['City'].value_counts() > 10]

# Identify Top 5 Cities with highest average rating
top_rated_cities = reliable_cities.sort_values(by='Aggregate rating', ascending=False).head(5)

# Identify Top 5 Most expensive cities (by Average Cost)
most_expensive_cities = reliable_cities.sort_values(by='Average Cost for two', ascending=False).head(5)

print("\n--- Insights: Top Rated Cities (min 10 restaurants) ---")
print(top_rated_cities[['Aggregate rating']])

print("\n--- Insights: Most Expensive Cities (min 10 restaurants) ---")
print(most_expensive_cities[['Average Cost for two']])