import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
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
df['sex_encoded'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked_encoded'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# ── Select features and target ────────────────────────────────────────
features = ['pclass', 'sex_encoded', 'age', 'fare',
            'family_size', 'is_alone', 'embarked_encoded']
X = df[features]
y = df['survived']

# ── Split into train and test sets ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ── Scale features ────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training set size :", X_train.shape)
print("Testing set size  :", X_test.shape)
print("Features used     :", features)
# ── Model 1: Logistic Regression ─────────────────────────────────────
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"Logistic Regression Accuracy : {lr_acc:.4f}")

# ── Model 2: Decision Tree ────────────────────────────────────────────
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
print(f"Decision Tree Accuracy       : {dt_acc:.4f}")

# ── Model 3: Random Forest ────────────────────────────────────────────
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy       : {rf_acc:.4f}")
# ── Grid Search — Tune Random Forest ─────────────────────────────────
print("\nRunning Grid Search — this may take a minute...")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best Parameters  : {grid_search.best_params_}")
print(f"Best CV Accuracy : {grid_search.best_score_:.4f}")

# ── Evaluate tuned model ──────────────────────────────────────────────
best_rf = grid_search.best_estimator_
best_rf_pred = best_rf.predict(X_test)
best_rf_acc = accuracy_score(y_test, best_rf_pred)
print(f"Tuned RF Accuracy: {best_rf_acc:.4f}")
# ── Cross Validation Scores ───────────────────────────────────────────
print("\nCross Validation Scores:")
lr_cv = cross_val_score(lr_model, X_train_scaled, y_train, cv=5)
rf_cv = cross_val_score(best_rf, X_train, y_train, cv=5)
dt_cv = cross_val_score(dt_model, X_train, y_train, cv=5)
print(f"Logistic Regression CV : {lr_cv.mean():.4f}")
print(f"Random Forest CV       : {rf_cv.mean():.4f}")
print(f"Decision Tree CV       : {dt_cv.mean():.4f}")

# ── Confusion Matrix — Best Model (Logistic Regression) ──────────────
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Did Not Survive', 'Survived'],
            yticklabels=['Did Not Survive', 'Survived'])
plt.title('Confusion Matrix — Logistic Regression')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('chart_confusion_matrix.png')
plt.show()

# ── ROC Curve ─────────────────────────────────────────────────────────
plt.figure(figsize=(7, 5))
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
rf_prob = best_rf.predict_proba(X_test)[:, 1]
dt_prob = dt_model.predict_proba(X_test)[:, 1]

for prob, name, color in zip(
    [lr_prob, rf_prob, dt_prob],
    ['Logistic Regression', 'Random Forest', 'Decision Tree'],
    ['blue', 'green', 'orange']
):
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('ROC Curve — All Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig('chart_roc_curve.png')
plt.show()

# ── Feature Importance — Random Forest ───────────────────────────────
plt.figure(figsize=(8, 5))
importance = pd.Series(best_rf.feature_importances_, index=features)
importance.sort_values().plot(kind='barh', color='steelblue')
plt.title('Feature Importance — Random Forest')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('chart_feature_importance.png')
plt.show()

# ── Model Accuracy Comparison ─────────────────────────────────────────
models = ['Logistic Regression', 'Decision Tree', 'Random Forest\n(Tuned)']
accuracies = [lr_acc, dt_acc, best_rf_acc]
plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=['blue', 'orange', 'green'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0.6, 0.85)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{acc:.4f}', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('chart_model_comparison.png')
plt.show()

# ── Classification Report ─────────────────────────────────────────────
print("\nClassification Report — Logistic Regression:")
print(classification_report(y_test, lr_pred,
      target_names=['Did Not Survive', 'Survived']))
      