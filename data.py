import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def main():
    try:
        # Task 1: Load and Explore Dataset
        iris = load_iris()
        df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                          columns= iris['feature_names'] + ['target'])
        df['species'] = df['target'].map({i:name for i, name in enumerate(iris['target_names'])})

        print("First 5 rows of the dataset:")
        print(df.head(), "\n")

        print("Data Types and Missing Values:")
        print(df.info(), "\n")

        
        if df.isnull().sum().any():
            print("Missing values found. Filling with column mean...")
            df.fillna(df.mean(), inplace=True)
        else:
            print("No missing values found.\n")

        # Task 2: Basic Data Analysis
        print("Basic Statistics of Numerical Columns:")
        print(df.describe(), "\n")

        print("Mean of numerical columns grouped by species:")
        grouped_mean = df.groupby('species').mean()
        print(grouped_mean, "\n")

        # Patterns / Insights
        print("Insights:")
        print("- Setosa tends to have smaller petal length and width compared to Versicolor and Virginica.")
        print("- Virginica has the largest petal length and sepal width on average.\n")

        # Task 3: Data Visualization
        sns.set(style="whitegrid")

        # 1. Line chart 
        plt.figure(figsize=(8,5))
        line_data = df.groupby('species')['sepal length (cm)'].mean()
        line_data.plot(marker='o')
        plt.title('Average Sepal Length by Species (Line Chart)')
        plt.xlabel('Species')
        plt.ylabel('Average Sepal Length (cm)')
        plt.grid(True)
        plt.show()

        # 2. Bar chart - average petal length per species
        plt.figure(figsize=(8,5))
        sns.barplot(x='species', y='petal length (cm)', data=df, palette='viridis')
        plt.title('Average Petal Length by Species (Bar Chart)')
        plt.xlabel('Species')
        plt.ylabel('Petal Length (cm)')
        plt.show()

        # 3. Histogram - distribution of sepal width
        plt.figure(figsize=(8,5))
        sns.histplot(df['sepal width (cm)'], bins=20, kde=True, color='blue')
        plt.title('Distribution of Sepal Width (Histogram)')
        plt.xlabel('Sepal Width (cm)')
        plt.ylabel('Frequency')
        plt.show()

        # 4. Scatter plot - sepal length vs petal length colored by species
        plt.figure(figsize=(8,5))
        sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, palette='deep')
        plt.title('Sepal Length vs Petal Length (Scatter Plot)')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        plt.legend(title='Species')
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()