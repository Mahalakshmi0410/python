import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = sns.load_dataset('titanic')

# Step 1: Basic inspection
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())
print("\nBasic Stats:\n", df.describe())
# Remove duplicates
print("Before removing duplicates:", df.shape)
df = df.drop_duplicates()
print("After removing duplicates:", df.shape)

# Handle missing values
# 1. Age — fill with median grouped by class and sex
df['age'] = df.groupby(['pclass', 'sex'])['age'].transform(
    lambda x: x.fillna(x.median())
)

# 2. Embarked — fill with mode (most common port)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# 3. Deck — too many missing, drop it
df.drop(columns=['deck'], inplace=True)

# Confirm results
print("\nMissing values after cleaning:")
print(df.isnull().sum())
print("\nShape after cleaning:", df.shape)
# Fix embarked - force fill with mode
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])
print(df.isnull().sum())
# Step 3: Feature Engineering

# 1. Family size — how many people travelling with them
df['family_size'] = df['sibsp'] + df['parch'] + 1

# 2. Is alone — travelling alone or not
df['is_alone'] = (df['family_size'] == 1).astype(int)

# 3. Age group — categorise age into bins
df['age_group'] = pd.cut(df['age'], 
                          bins=[0, 12, 17, 59, 100], 
                          labels=['Child', 'Teen', 'Adult', 'Senior'])

# 4. Fare category — bin fare into low, medium, high
df['fare_category'] = pd.cut(df['fare'],
                              bins=[0, 10, 50, 600],
                              labels=['Low', 'Medium', 'High'])

# 5. Encode sex column — male=0, female=1
df['sex_encoded'] = df['sex'].map({'male': 0, 'female': 1})

# Confirm new columns
print("New shape:", df.shape)
print("\nNew columns added:")
print(df[['family_size', 'is_alone', 'age_group', 'fare_category', 'sex_encoded']].head(10))
# Chart 1: Family size distribution
df['family_size'].value_counts().sort_index().plot(kind='bar', color='steelblue')
plt.title('Family Size Distribution')
plt.xlabel('Family Size')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('chart6_family_size.png')
plt.show()

# Chart 2: Survival rate by age group
df.groupby('age_group', observed=True)['survived'].mean().plot(kind='bar', color='coral')
plt.title('Survival Rate by Age Group')
plt.ylabel('Survival Rate')
plt.tight_layout()
plt.savefig('chart7_survival_by_agegroup.png')
plt.show()

# Chart 3: Boxplot of fare by passenger class (outlier detection)
df.boxplot(column='fare', by='pclass', figsize=(8,5))
plt.title('Fare Distribution by Passenger Class')
plt.suptitle('')
plt.ylabel('Fare')
plt.tight_layout()
plt.savefig('chart8_fare_boxplot.png')
plt.show()