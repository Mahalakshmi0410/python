import seaborn as sns
import pandas as pd
import numpy as np

# Load the dataset directly — no download needed
df = sns.load_dataset('titanic')

print(df.shape)   # shows rows and columns
print(df.head())  # shows first 5 rows
# Basic info
print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())
import matplotlib.pyplot as plt

# Chart 1: Survival Count
df['survived'].value_counts().plot(kind='bar', color=['red','green'])
plt.title('Survival Count (0=Died, 1=Survived)')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('chart1_survival_count.png')
plt.show()

# Chart 2: Age Distribution
df['age'].dropna().hist(bins=30, color='steelblue')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('chart2_age_distribution.png')
plt.show()

# Chart 3: Survival Rate by Gender
df.groupby('sex')['survived'].mean().plot(kind='bar', color=['coral','skyblue'])
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')
plt.tight_layout()
plt.savefig('chart3_survival_by_gender.png')
plt.show()

# Chart 4: Survival Rate by Passenger Class
df.groupby('pclass')['survived'].mean().plot(kind='bar', color=['gold','silver','brown'])
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.tight_layout()
plt.savefig('chart4_survival_by_class.png')
plt.show()

# Chart 5: Correlation Heatmap
import seaborn as sns
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('chart5_correlation_heatmap.png')
plt.show()