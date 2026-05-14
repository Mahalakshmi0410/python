import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Reload and clean dataset ──────────────────────────────────────────
df = sns.load_dataset('titanic')
df = df.drop_duplicates()
df['age'] = df.groupby(['pclass', 'sex'])['age'].transform(lambda x: x.fillna(x.median()))
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])
df.drop(columns=['deck'], inplace=True)
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['is_alone'] = (df['family_size'] == 1).astype(int)
df['age_group'] = pd.cut(df['age'], bins=[0,12,17,59,100], labels=['Child','Teen','Adult','Senior'])
df['sex_encoded'] = df['sex'].map({'male': 0, 'female': 1})

print("Dataset ready:", df.shape)
# ══════════════════════════════════════════════════════
# HYPOTHESIS TEST 1: Chi-Square — Gender vs Survival
# ══════════════════════════════════════════════════════
print("=" * 55)
print("TEST 1: Chi-Square Test — Gender vs Survival")
print("=" * 55)

contingency_gender = pd.crosstab(df['sex'], df['survived'])
print("\nContingency Table:")
print(contingency_gender)

chi2_gender, p_gender, dof_gender, expected_gender = stats.chi2_contingency(contingency_gender)
print(f"\nChi-Square Statistic : {chi2_gender:.4f}")
print(f"P-Value              : {p_gender:.6f}")
print(f"Degrees of Freedom   : {dof_gender}")
if p_gender < 0.05:
    print("Result: REJECT null hypothesis — Gender significantly affects survival")
else:
    print("Result: FAIL TO REJECT null hypothesis")

# ══════════════════════════════════════════════════════
# HYPOTHESIS TEST 2: Chi-Square — Pclass vs Survival
# ══════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("TEST 2: Chi-Square Test — Passenger Class vs Survival")
print("=" * 55)

contingency_class = pd.crosstab(df['pclass'], df['survived'])
print("\nContingency Table:")
print(contingency_class)

chi2_class, p_class, dof_class, expected_class = stats.chi2_contingency(contingency_class)
print(f"\nChi-Square Statistic : {chi2_class:.4f}")
print(f"P-Value              : {p_class:.6f}")
print(f"Degrees of Freedom   : {dof_class}")
if p_class < 0.05:
    print("Result: REJECT null hypothesis — Passenger class significantly affects survival")
else:
    print("Result: FAIL TO REJECT null hypothesis")
# ══════════════════════════════════════════════════════
# HYPOTHESIS TEST 3: T-Test — Age vs Survival
# ══════════════════════════════════════════════════════
print("=" * 55)
print("TEST 3: T-Test — Age vs Survival")
print("=" * 55)

survived_age = df[df['survived'] == 1]['age']
not_survived_age = df[df['survived'] == 0]['age']

print(f"\nAverage age of survivors     : {survived_age.mean():.2f}")
print(f"Average age of non-survivors : {not_survived_age.mean():.2f}")

t_stat_age, p_age = stats.ttest_ind(survived_age, not_survived_age)
print(f"\nT-Statistic : {t_stat_age:.4f}")
print(f"P-Value     : {p_age:.6f}")
if p_age < 0.05:
    print("Result: REJECT null hypothesis — Age significantly differs between survivors and non-survivors")
else:
    print("Result: FAIL TO REJECT null hypothesis — No significant age difference")

# ══════════════════════════════════════════════════════
# HYPOTHESIS TEST 4: T-Test — Fare vs Survival
# ══════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("TEST 4: T-Test — Fare vs Survival")
print("=" * 55)

survived_fare = df[df['survived'] == 1]['fare']
not_survived_fare = df[df['survived'] == 0]['fare']

print(f"\nAverage fare of survivors     : {survived_fare.mean():.2f}")
print(f"Average fare of non-survivors : {not_survived_fare.mean():.2f}")

t_stat_fare, p_fare = stats.ttest_ind(survived_fare, not_survived_fare)
print(f"\nT-Statistic : {t_stat_fare:.4f}")
print(f"P-Value     : {p_fare:.6f}")
if p_fare < 0.05:
    print("Result: REJECT null hypothesis — Fare significantly differs between survivors and non-survivors")
else:
    print("Result: FAIL TO REJECT null hypothesis — No significant fare difference")
# ══════════════════════════════════════════════════════
# HYPOTHESIS TEST 5: Correlation Analysis
# ══════════════════════════════════════════════════════
print("=" * 55)
print("TEST 5: Correlation Analysis — Variables vs Survival")
print("=" * 55)

numeric_cols = ['pclass', 'age', 'fare', 'family_size', 'is_alone', 'sex_encoded']
correlations = df[numeric_cols].corrwith(df['survived']).sort_values(ascending=False)

print("\nCorrelation with Survival:")
print(correlations.round(4))

# ── Visualisation 1: Correlation Bar Chart ────────────
correlations.plot(kind='bar', color=['green' if x > 0 else 'red' for x in correlations],
                  figsize=(8, 5))
plt.title('Correlation of Variables with Survival')
plt.ylabel('Correlation Coefficient')
plt.xlabel('Variable')
plt.axhline(y=0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('chart_correlation_bar.png')
plt.show()

# ── Visualisation 2: Confusion-style survival by gender ──
plt.figure(figsize=(6, 4))
sns.heatmap(pd.crosstab(df['sex'], df['survived'], normalize='index').round(2),
            annot=True, fmt='.2f', cmap='RdYlGn')
plt.title('Survival Rate by Gender (Proportion)')
plt.xlabel('Survived (0=No, 1=Yes)')
plt.tight_layout()
plt.savefig('chart_gender_heatmap.png')
plt.show()

# ── Visualisation 3: Age distribution by survival ────────
plt.figure(figsize=(8, 5))
df[df['survived']==1]['age'].hist(bins=25, alpha=0.6, color='green', label='Survived')
df[df['survived']==0]['age'].hist(bins=25, alpha=0.6, color='red', label='Did Not Survive')
plt.title('Age Distribution by Survival Status')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('chart_age_distribution.png')
plt.show()