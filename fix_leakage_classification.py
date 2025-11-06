#!/usr/bin/env python3
"""
Διόρθωση Data Leakage - Classification χωρίς Leakage
Χρησιμοποιεί lag features (δεδομένα από προηγούμενες μέρες)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CLASSIFICATION ΧΩΡΙΣ DATA LEAKAGE")
print("="*70)

# Φόρτωση δεδομένων
df = pd.read_csv('data/household_power_daily_clean.csv', index_col=0, parse_dates=True)
print(f"\nΑρχικά δεδομένα: {df.shape}")

# Δημιουργία target
mean_consumption = df['Daily_total_power'].mean()
df['High_Consumption'] = (df['Daily_total_power'] > mean_consumption).astype(int)

print("\n" + "="*70)
print("ΔΗΜΙΟΥΡΓΙΑ LAG FEATURES (χωρίς leakage)")
print("="*70)

# ===== LAG FEATURES: Χρησιμοποιούμε δεδομένα από ΠΡΟΗΓΟΥΜΕΝΕΣ μέρες =====

# 1. Lag features από προηγούμενες μέρες
df['total_power_lag1'] = df['Daily_total_power'].shift(1)     # χθες
df['total_power_lag2'] = df['Daily_total_power'].shift(2)     # προχθές
df['total_power_lag7'] = df['Daily_total_power'].shift(7)     # πριν 1 εβδομάδα

df['peak_power_lag1'] = df['Peak_hour_power'].shift(1)
df['nighttime_lag1'] = df['Nighttime_usage'].shift(1)

df['high_consumption_lag1'] = df['High_Consumption'].shift(1)  # Ήταν υψηλή χθες;
df['high_consumption_lag7'] = df['High_Consumption'].shift(7)  # Πριν 1 εβδομάδα;

# 2. Rolling statistics (χωρίς τη σημερινή μέρα!)
df['rolling_mean_7d'] = df['Daily_total_power'].shift(1).rolling(window=7, min_periods=3).mean()
df['rolling_std_7d'] = df['Daily_total_power'].shift(1).rolling(window=7, min_periods=3).std()
df['rolling_max_7d'] = df['Daily_total_power'].shift(1).rolling(window=7, min_periods=3).max()
df['rolling_min_7d'] = df['Daily_total_power'].shift(1).rolling(window=7, min_periods=3).min()

# 3. Trend features
df['power_diff_1d'] = df['Daily_total_power'].shift(1) - df['Daily_total_power'].shift(2)  # Αύξηση χθες vs προχθές
df['power_ratio_lag1_lag7'] = df['total_power_lag1'] / (df['total_power_lag7'] + 0.001)  # Λόγος χθες / πριν 1 εβδομάδα

# 4. Day-of-week patterns (από το παρελθόν)
df['same_weekday_avg_lag'] = df.groupby('DayOfWeek')['Daily_total_power'].transform(
    lambda x: x.shift(1).expanding().mean()
)

# 5. Χρονικά features (ΜΟΝΟ αυτά δεν έχουν leakage!)
temporal_features = ['DayOfWeek', 'IsWeekend', 'Month', 'DayOfYear']

print("\nLag features που δημιουργήθηκαν:")
lag_features = [col for col in df.columns if 'lag' in col or 'rolling' in col or 'diff' in col or 'ratio' in col]
for feat in lag_features:
    print(f"  - {feat}")

# ===== FEATURE SELECTION =====
features_to_use = lag_features + temporal_features + ['same_weekday_avg_lag']

X = df[features_to_use].copy()
y = df['High_Consumption'].copy()

# Αφαίρεση rows με NaN (από τα shifts)
valid_idx = ~X.isnull().any(axis=1)
X = X[valid_idx]
y = y[valid_idx]

print(f"\nΔεδομένα μετά την αφαίρεση NaN: {X.shape}")
print(f"Features που χρησιμοποιούνται: {len(X.columns)}")

# ===== TRAIN/TEST SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ===== TRAINING MODELS =====
print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

results = []

for name, model in models.items():
    print(f"\n{name}...")
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    results.append({
        'Model': name,
        'Test Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'CV Accuracy': cv_scores.mean(),
        'CV Std': cv_scores.std()
    })
    
    print(f"  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ===== RESULTS =====
print("\n" + "="*70)
print("ΑΠΟΤΕΛΕΣΜΑΤΑ (χωρίς leakage)")
print("="*70)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Σύγκριση με τα "παλιά" αποτελέσματα
print("\n" + "="*70)
print("ΣΥΓΚΡΙΣΗ:")
print("="*70)
print("ΜΕ LEAKAGE (πριν):     Random Forest 100.00% | SVM 95.85% | LogReg 99.31%")
best_acc = results_df['Test Accuracy'].max()
print(f"ΧΩΡΙΣ LEAKAGE (τώρα): Best Model {best_acc*100:.2f}%")
print("\nΗ μείωση accuracy είναι ΦΥΣΙΟΛΟΓΙΚΗ και ΑΝΑΜΕΝΟΜΕΝΗ!")
print("Τώρα το μοντέλο μαθαίνει πραγματικά πρότυπα, όχι shortcuts.")

# ===== VISUALIZATION =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy comparison
axes[0].bar(results_df['Model'], results_df['Test Accuracy'], color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.7, edgecolor='black')
axes[0].axhline(y=0.85, color='red', linestyle='--', linewidth=2, label='Target (85%)')
axes[0].set_ylabel('Test Accuracy')
axes[0].set_title('Classification Accuracy (ΧΩΡΙΣ Leakage)')
axes[0].set_ylim([0.5, 1.0])
axes[0].legend()
axes[0].grid(True, axis='y', alpha=0.3)

for i, (model, acc) in enumerate(zip(results_df['Model'], results_df['Test Accuracy'])):
    axes[0].text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontweight='bold')

# Plot 2: CV scores
axes[1].bar(results_df['Model'], results_df['CV Accuracy'], 
            yerr=results_df['CV Std'], color=['#e74c3c', '#3498db', '#2ecc71'], 
            alpha=0.7, edgecolor='black', capsize=10)
axes[1].axhline(y=0.85, color='red', linestyle='--', linewidth=2, label='Target (85%)')
axes[1].set_ylabel('CV Accuracy')
axes[1].set_title('Cross-Validation Accuracy (5-Fold)')
axes[1].set_ylim([0.5, 1.0])
axes[1].legend()
axes[1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/classification_no_leakage_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Γράφημα αποθηκεύτηκε: results/classification_no_leakage_comparison.png")

# ===== ΤΕΛΙΚΟ ΜΗΝΥΜΑ =====
print("\n" + "="*70)
print("ΣΥΜΠΕΡΑΣΜΑ")
print("="*70)

if best_acc > 0.85:
    print(f"✅ Ξεπεράσαμε τον στόχο 85% ({best_acc*100:.2f}%) χωρίς leakage!")
elif best_acc > 0.80:
    print(f"⚠️  Κοντά στον στόχο ({best_acc*100:.2f}%), αλλά ΧΩΡΙΣ leakage!")
    print("   Αυτό είναι πολύ πιο έγκυρο από το 100% με leakage.")
else:
    print(f"⚠️  Κάτω από τον στόχο ({best_acc*100:.2f}%)")
    print("   Χρειάζονται περισσότερα features ή tuning.")

print("\nΑυτά τα αποτελέσματα είναι ΑΞΙΟΠΙΣΤΑ και ΧΡΗΣΙΜΑ στην πράξη!")
print("="*70)
