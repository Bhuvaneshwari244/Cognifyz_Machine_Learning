import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
try:
    df = pd.read_csv('Dataset .csv')
except FileNotFoundError:
    # Update this path if necessary
    df = pd.read_csv(r'C:\Users\vijay kumar\OneDrive\Desktop\Cognifyz_machine_learning\Aggregate rating\Dataset .csv')

# --- GRAPH 1: Geographical Distribution ---
plt.figure(figsize=(10, 6))
# Using scatterplot to visualize distribution
sns.scatterplot(x='Longitude', y='Latitude', hue='Country Code', palette='viridis', data=df, alpha=0.6, legend=False)
plt.title('Geographical Distribution of Restaurants')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)

# Save Graph 1
plt.savefig('task4_geo_distribution.png')
print("1. Map visualization saved as 'task4_geo_distribution.png'")

# Clear the plot to avoid overlapping with the next one
plt.clf() 

# --- GRAPH 2: City Concentration ---
city_counts = df['City'].value_counts()
top_cities = city_counts.head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_cities.index, y=top_cities.values, palette='magma')
plt.title('Top 10 Cities with Highest Concentration of Restaurants')
plt.xlabel('City')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=45)
plt.tight_layout()

# Save Graph 2
plt.savefig('task4_city_concentration.png')
print("2. City concentration chart saved as 'task4_city_concentration.png'")

# Display charts (Optional - code will finish even if you don't close immediately if placed at end)
plt.show()

# --- STATISTICS ---
print("\nGenerating Statistics...")
city_stats = df.groupby('City')[['Aggregate rating', 'Average Cost for two']].mean()
reliable_cities = city_stats[df['City'].value_counts() > 10]

top_rated = reliable_cities.sort_values(by='Aggregate rating', ascending=False).head(5)
print("\nTop 5 Rated Cities:\n", top_rated['Aggregate rating'])