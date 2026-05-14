import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reload and re-clean the dataset (same steps as Week 2)
df = sns.load_dataset('titanic')
df = df.drop_duplicates()
df['age'] = df.groupby(['pclass', 'sex'])['age'].transform(lambda x: x.fillna(x.median()))
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])
df.drop(columns=['deck'], inplace=True)

# Feature engineering
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['is_alone'] = (df['family_size'] == 1).astype(int)
df['age_group'] = pd.cut(df['age'], bins=[0, 12, 17, 59, 100], labels=['Child', 'Teen', 'Adult', 'Senior'])
df['fare_category'] = pd.cut(df['fare'], bins=[0, 10, 50, 600], labels=['Low', 'Medium', 'High'])
df['sex_encoded'] = df['sex'].map({'male': 0, 'female': 1})

print("Dataset ready:", df.shape)
# ── Chart 1: Bar Chart — Survival Rate by Gender and Class ──────────
pivot = df.groupby(['pclass', 'sex'])['survived'].mean().unstack()
pivot.plot(kind='bar', color=['coral', 'steelblue'], figsize=(8, 5))
plt.title('Survival Rate by Passenger Class and Gender')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.xticks([0, 1, 2], ['First Class', 'Second Class', 'Third Class'], rotation=0)
plt.legend(['Female', 'Male'])
plt.tight_layout()
plt.savefig('chart_bar.png')
plt.show()

# ── Chart 2: Line Chart — Average Age by Class and Survival ─────────
line_data = df.groupby(['pclass', 'survived'])['age'].mean().unstack()
line_data.plot(kind='line', marker='o', figsize=(8, 5), color=['red', 'green'])
plt.title('Average Age by Passenger Class and Survival Status')
plt.xlabel('Passenger Class')
plt.ylabel('Average Age')
plt.xticks([0, 1, 2], ['First Class', 'Second Class', 'Third Class'])
plt.legend(['Did Not Survive', 'Survived'])
plt.tight_layout()
plt.savefig('chart_line.png')
plt.show()

# ── Chart 3: Heatmap — Correlation Matrix ───────────────────────────
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f',
            cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numeric Variables')
plt.tight_layout()
plt.savefig('chart_heatmap.png')
plt.show()
# ── Chart 4: Scatter Plot — Age vs Fare coloured by Survival ────────
plt.figure(figsize=(9, 5))
colors = df['survived'].map({0: 'red', 1: 'green'})
plt.scatter(df['age'], df['fare'], c=colors, alpha=0.5, edgecolors='none')
plt.title('Age vs Fare — Coloured by Survival')
plt.xlabel('Age')
plt.ylabel('Fare Paid')
plt.tight_layout()
plt.savefig('chart_scatter.png')
plt.show()
# ── Chart 5: Bar Chart — Survival Count by Embarkation Port ─────────
sns.countplot(data=df, x='embarked', hue='survived',
              palette={0: 'red', 1: 'green'})
plt.title('Survival Count by Embarkation Port')
plt.xlabel('Port of Embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)')
plt.ylabel('Number of Passengers')
plt.legend(['Did Not Survive', 'Survived'])
plt.tight_layout()
plt.savefig('chart_embark.png')
plt.show()

# ── Chart 6: Heatmap — Survival Rate by Class and Age Group ─────────
pivot2 = df.groupby(['pclass', 'age_group'], observed=True)['survived'].mean().unstack()
plt.figure(figsize=(8, 5))
sns.heatmap(pivot2, annot=True, fmt='.2f', cmap='RdYlGn',
            linewidths=0.5, vmin=0, vmax=1)
plt.title('Survival Rate by Passenger Class and Age Group')
plt.xlabel('Age Group')
plt.ylabel('Passenger Class')
plt.tight_layout()
plt.savefig('chart_class_age_heatmap.png')
plt.show()