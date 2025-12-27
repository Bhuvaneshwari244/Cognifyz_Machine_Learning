import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
try:
    df = pd.read_csv('Dataset .csv')
except FileNotFoundError:
    df = pd.read_csv(r'C:\Users\vijay kumar\OneDrive\Desktop\Cognifyz_machine_learning\Aggregate rating\Dataset .csv')

# --- PREPARE DATA ---
# Data for Graph 2 (Calculated before plotting to keep logic clean)
city_counts = df['City'].value_counts()
top_cities = city_counts.head(10)


# --- GRAPH 1: Geographical Distribution ---
# Create Figure 1 (Window 1)
plt.figure(1, figsize=(10, 6)) 

sns.scatterplot(x='Longitude', y='Latitude', hue='Country Code', palette='viridis', data=df, alpha=0.6, legend=False)
plt.title('Geographical Distribution of Restaurants')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)

# Save Graph 1
plt.savefig('task4_geo_distribution.png')
print("Graph 1 ready (and saved).")


# --- GRAPH 2: City Concentration ---
# Create Figure 2 (Window 2)
plt.figure(2, figsize=(10, 6)) 

sns.barplot(x=top_cities.index, y=top_cities.values, hue=top_cities.index, palette='magma', legend=False)
plt.title('Top 10 Cities with Highest Concentration of Restaurants')
plt.xlabel('City')
plt.ylabel('Number of Restaurants')
plt.xticks(rotation=45)
plt.tight_layout()

# Save Graph 2
plt.savefig('task4_city_concentration.png')
print("Graph 2 ready (and saved).")


# --- DISPLAY BOTH ---
print("Opening both graphs now...")
# This command opens ALL active figures (Figure 1 and Figure 2) at the same time
plt.show()

print("Closed.")