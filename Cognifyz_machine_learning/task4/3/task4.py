import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
try:
    df = pd.read_csv('Dataset .csv')
except FileNotFoundError:
    df = pd.read_csv(r'C:\Users\vijay kumar\OneDrive\Desktop\Cognifyz_machine_learning\Aggregate rating\Dataset .csv')

# --- GRAPH 1: Geographical Distribution ---
# Create a dedicated figure for the first plot
plt.figure(figsize=(10, 6))

# Create the scatter plot
sns.scatterplot(x='Longitude', y='Latitude', hue='Country Code', palette='viridis', data=df, alpha=0.6, legend=False)

plt.title('Geographical Distribution of Restaurants')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)

# Save and CLOSE immediately to prevent empty files
plt.savefig('task4_geo_distribution.png')
plt.close() # <--- This ensures the file is written and memory is cleared
print("1. Map visualization saved as 'task4_geo_distribution.png'")


# --- GRAPH 2: City Concentration ---
# Create a new figure for the second plot
plt.figure(figsize=(10, 6))

city_counts = df['City'].value_counts()
top_cities = city_counts.head(10)

# FIXED: Added hue=top_cities.index and legend=False to fix the warning
sns.barplot(x=top_cities.index, y=top_cities.values, hue=top_cities.index, palette='magma', legend=False)

plt.title('Top 10 Cities with Highest Concentration of Restaurants')
plt.xlabel('City')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=45)
plt.tight_layout()

# Save and CLOSE
plt.savefig('task4_city_concentration.png')
plt.close() # <--- Ensures this file is also safe
print("2. City concentration chart saved as 'task4_city_concentration.png'")

print("\nAll tasks completed successfully. Check your folder for the images.")