#!/usr/bin/env python3
"""
Έλεγχος για Data Leakage στο Classification Model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Φόρτωση δεδομένων
df = pd.read_csv('data/household_power_daily_clean.csv', index_col=0, parse_dates=True)

# Δημιουργία target
mean_consumption = df['Daily_total_power'].mean()
df['High_Consumption'] = (df['Daily_total_power'] > mean_consumption).astype(int)

print("="*70)
print("ΕΛΕΓΧΟΣ ΓΙΑ DATA LEAKAGE")
print("="*70)

# Test 1: Έλεγχος αν τα features περιέχουν το target
print("\n1. Έλεγχος σχέσης features με target...")
print(f"   Target: High_Consumption (βασίζεται στο Daily_total_power)")

# Υπολογισμός συσχέτισης
features_to_check = [
    'Daily_mean_power',
    'Peak_hour_power', 
    'Nighttime_usage',
    'Morning_usage',
    'Afternoon_usage',
    'Global_intensity_mean',
    'Daily_total_power'  # Το ίδιο το target!
]

# Correlation με το target
correlations = []
for feature in features_to_check:
    corr = df[feature].corr(df['Daily_total_power'])
    correlations.append({
        'Feature': feature,
        'Correlation with Daily_total_power': corr
    })

corr_df = pd.DataFrame(correlations).sort_values('Correlation with Daily_total_power', ascending=False)

print("\nΣυσχέτιση features με Daily_total_power:")
print(corr_df.to_string(index=False))

# Test 2: Έλεγχος αν οι περίοδοι ημέρας αθροίζουν στο total
print("\n" + "="*70)
print("2. Έλεγχος αν τα time-period features αθροίζουν στο total...")
print("="*70)

df['sum_periods'] = (df['Peak_hour_power'] + df['Nighttime_usage'] + 
                     df['Morning_usage'] + df['Afternoon_usage'])
df['difference'] = abs(df['Daily_total_power'] - df['sum_periods'])

print(f"\nΜέση διαφορά: {df['difference'].mean():.4f} kWh")
print(f"Max διαφορά: {df['difference'].max():.4f} kWh")
print(f"% γραμμών με διαφορά < 0.1 kWh: {(df['difference'] < 0.1).sum() / len(df) * 100:.2f}%")

if df['difference'].mean() < 1.0:
    print("\n⚠️  ΠΡΟΣΟΧΗ: Τα time-period features αθροίζουν (περίπου) στο Daily_total_power!")
    print("   Αυτό είναι LEAKAGE!")

# Test 3: Οπτικοποίηση
print("\n" + "="*70)
print("3. Δημιουργία οπτικοποιήσεων...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Daily_mean_power vs Daily_total_power (χρωματισμένο με High_Consumption)
axes[0, 0].scatter(df['Daily_mean_power'], df['Daily_total_power'], 
                   c=df['High_Consumption'], cmap='coolwarm', alpha=0.6)
axes[0, 0].set_xlabel('Daily_mean_power')
axes[0, 0].set_ylabel('Daily_total_power (TARGET)')
axes[0, 0].set_title('LEAKAGE: Daily_mean_power vs Target')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Sum of periods vs Daily_total_power
axes[0, 1].scatter(df['sum_periods'], df['Daily_total_power'], alpha=0.6)
axes[0, 1].plot([0, 80], [0, 80], 'r--', linewidth=2, label='Perfect match')
axes[0, 1].set_xlabel('Sum(Peak + Night + Morning + Afternoon)')
axes[0, 1].set_ylabel('Daily_total_power (TARGET)')
axes[0, 1].set_title('LEAKAGE: Time Periods Sum vs Target')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Correlation heatmap
corr_matrix = df[features_to_check + ['High_Consumption']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            ax=axes[1, 0], cbar_kws={'label': 'Correlation'})
axes[1, 0].set_title('Correlation Matrix (με Target)')

# Plot 4: Distribution of differences
axes[1, 1].hist(df['difference'], bins=50, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0.1, color='red', linestyle='--', linewidth=2, 
                   label='Threshold (0.1 kWh)')
axes[1, 1].set_xlabel('|Daily_total_power - Sum of Periods|')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Διαφορά Total vs Sum of Periods')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/data_leakage_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Γράφημα αποθηκεύτηκε: results/data_leakage_analysis.png")

# ΤΕΛΙΚΗ ΔΙΑΓΝΩΣΗ
print("\n" + "="*70)
print("ΔΙΑΓΝΩΣΗ")
print("="*70)

high_corr_features = corr_df[corr_df['Correlation with Daily_total_power'] > 0.9]
print(f"\nFeatures με συσχέτιση > 0.9: {len(high_corr_features)}")
print(high_corr_features.to_string(index=False))

print("\n🔴 ΣΥΜΠΕΡΑΣΜΑ:")
if len(high_corr_features) > 2:  # Daily_total_power + άλλα
    print("   ΝΑΙ, ΥΠΑΡΧΕΙ DATA LEAKAGE!")
    print("   Χρησιμοποιείς features που είναι μέρος ή άμεσα υπολογίζονται από το target.")
    print("\n   ΛΥΣΗ:")
    print("   - Χρησιμοποίησε lag features (δεδομένα από προηγούμενες μέρες)")
    print("   - Ή πρόβλεψε την επόμενη μέρα αντί για τη σημερινή")
else:
    print("   Δεν βρέθηκε σοβαρό leakage.")

print("="*70)
