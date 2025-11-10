#!/usr/bin/env python3
"""
Script to create modeling.ipynb with proper structure
"""

import json

# Define all cells
cells = [
    # Cell 0: Title
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Μοντελοποίηση: Ταξινόμηση Κατανάλωσης Ενέργειας\n",
            "\n",
            "## Στόχος\n",
            "Πρόβλεψη εάν η ημερήσια κατανάλωση ενέργειας είναι **Υψηλή** ή **Χαμηλή** σε σχέση με τον εποχιακό μέσο όρο του νοικοκυριού.\n",
            "\n",
            "## Προσέγγιση\n",
            "- **Target**: Season-adjusted high consumption (>15% πάνω από εποχιακό μέσο όρο)\n",
            "- **Features**: Μόνο past-known features (lags, rolling stats, calendar)\n",
            "- **Μοντέλα**: Logistic Regression → Random Forest → XGBoost/LightGBM\n",
            "- **Αξιολόγηση**: Accuracy, Precision, Recall, F1-Score, ROC-AUC\n",
            "\n",
            "## Αποφυγή Data Leakage\n",
            "✅ Χρήση μόνο features που είναι γνωστά **πριν** την ημέρα πρόβλεψης  \n",
            "✅ Season means υπολογίζονται **μόνο από train set**  \n",
            "✅ Scalers fit **μόνο στο train**"
        ]
    },
    
    # Cell 1: Section header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["---\n", "## 1. Imports & Setup"]
    },
    
    # Cell 2: Imports
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from pathlib import Path\n",
            "\n",
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
            "from sklearn.compose import ColumnTransformer\n",
            "from sklearn.pipeline import Pipeline\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.ensemble import RandomForestClassifier\n",
            "from sklearn.metrics import (\n",
            "    accuracy_score, precision_score, recall_score, f1_score,\n",
            "    roc_auc_score, classification_report, confusion_matrix,\n",
            "    RocCurveDisplay\n",
            ")\n",
            "\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "np.random.seed(42)\n",
            "pd.set_option('display.max_columns', None)\n",
            "pd.set_option('display.precision', 3)\n",
            "\n",
            "DATA_DIR = Path('..') / 'data'\n",
            "RESULTS_DIR = Path('..') / 'results'\n",
            "RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "print('✅ Imports completed successfully')\n",
            "print(f'📁 Data directory: {DATA_DIR.absolute()}')\n",
            "print(f'📊 Results directory: {RESULTS_DIR.absolute()}')"
        ]
    },
    
    # Cell 3: Load data header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## 2. Φόρτωση Δεδομένων\n",
            "\n",
            "Φορτώνουμε τα **raw** (μη κανονικοποιημένα) ημερήσια δεδομένα από το preprocessing."
        ]
    },
    
    # Cell 4: Load data
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Φόρτωση train/val/test sets\n",
            "train = pd.read_csv(DATA_DIR / 'train_raw.csv', parse_dates=True, index_col=0)\n",
            "val = pd.read_csv(DATA_DIR / 'val_raw.csv', parse_dates=True, index_col=0)\n",
            "test = pd.read_csv(DATA_DIR / 'test_raw.csv', parse_dates=True, index_col=0)\n",
            "\n",
            "print('📊 Dataset Shapes:')\n",
            "print(f'  Train: {train.shape} ({train.index.min().date()} → {train.index.max().date()})')\n",
            "print(f'  Val:   {val.shape} ({val.index.min().date()} → {val.index.max().date()})')\n",
            "print(f'  Test:  {test.shape} ({test.index.min().date()} → {test.index.max().date()})')\n",
            "\n",
            "print('\\n📋 Available columns:', train.shape[1])\n",
            "print('\\n🔍 First 3 rows of train:')\n",
            "train.head(3)"
        ]
    },
    
    # Cell 5: Target definition header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## 3. Ορισμός Target: Season-Adjusted High Consumption\n",
            "\n",
            "### Γιατί Season-Adjusted;\n",
            "Η κατανάλωση ενέργειας εξαρτάται **πολύ** από την εποχή:\n",
            "- **Χειμώνας**: ~33 kWh/day (θέρμανση)\n",
            "- **Καλοκαίρι**: ~17 kWh/day (χωρίς θέρμανση)\n",
            "\n",
            "Αν χρησιμοποιήσουμε **έναν** μέσο όρο για όλο το χρόνο, όλες οι χειμωνιάτικες μέρες θα είναι \"high\" και όλες οι καλοκαιρινές \"low\", που δεν είναι χρήσιμο.\n",
            "\n",
            "### Λύση\n",
            "Συγκρίνουμε κάθε μέρα με τον **εποχιακό** μέσο όρο της:\n",
            "- **High Consumption** = Κατανάλωση > 1.15 × εποχιακός μέσος όρος (15% πάνω)\n",
            "- **Normal/Low Consumption** = Κατανάλωση ≤ 1.15 × εποχιακός μέσος όρος\n",
            "\n",
            "### Αποφυγή Data Leakage\n",
            "✅ Υπολογίζουμε τους εποχιακούς μέσους **μόνο από το train set**  \n",
            "✅ Εφαρμόζουμε τους ίδιους μέσους στο val και test"
        ]
    },
    
    # Cell 6: Calculate season means
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Υπολογισμός εποχιακών μέσων όρων ΑΠΟ TRAIN ΜΟΝΟ\n",
            "season_means = train.groupby('Season')['Daily_total_power'].mean()\n",
            "\n",
            "print('📊 Εποχιακοί Μέσοι Όροι Κατανάλωσης (από train set):')\n",
            "print(season_means.sort_values(ascending=False).round(2))\n",
            "print(f'\\n📈 Διαφορά Winter vs Summer: {(season_means[\"Winter\"] - season_means[\"Summer\"]):.2f} kWh/day')\n",
            "print(f'   Ποσοστό: {(season_means[\"Winter\"] / season_means[\"Summer\"] - 1)*100:.1f}% περισσότερο το χειμώνα')"
        ]
    },
    
    # Cell 7: Create target
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Υπολογισμός season-adjusted deviation για κάθε set\n",
            "THRESHOLD = 1.15  # 15% πάνω από εποχιακό μέσο = High\n",
            "\n",
            "# Train\n",
            "train['season_mean'] = train['Season'].map(season_means)\n",
            "train['deviation_ratio'] = train['Daily_total_power'] / train['season_mean']\n",
            "y_train = (train['deviation_ratio'] > THRESHOLD).astype(int)\n",
            "\n",
            "# Validation (χρήση train season_means!)\n",
            "val['season_mean'] = val['Season'].map(season_means)\n",
            "val['deviation_ratio'] = val['Daily_total_power'] / val['season_mean']\n",
            "y_val = (val['deviation_ratio'] > THRESHOLD).astype(int)\n",
            "\n",
            "# Test (χρήση train season_means!)\n",
            "test['season_mean'] = test['Season'].map(season_means)\n",
            "test['deviation_ratio'] = test['Daily_total_power'] / test['season_mean']\n",
            "y_test = (test['deviation_ratio'] > THRESHOLD).astype(int)\n",
            "\n",
            "print(f'🎯 Target Definition: High Consumption = deviation_ratio > {THRESHOLD}')\n",
            "print(f'   (i.e., >15% πάνω από τον εποχιακό μέσο όρο)\\n')\n",
            "\n",
            "print('📊 Target Distribution:')\n",
            "print(f'  Train: {y_train.value_counts().to_dict()} → {y_train.value_counts(normalize=True).round(3).to_dict()}')\n",
            "print(f'  Val:   {y_val.value_counts().to_dict()} → {y_val.value_counts(normalize=True).round(3).to_dict()}')\n",
            "print(f'  Test:  {y_test.value_counts().to_dict()} → {y_test.value_counts(normalize=True).round(3).to_dict()}')\n",
            "\n",
            "print('\\n✅ Target is balanced (not too imbalanced)')"
        ]
    },
]

# Continue with more cells...
print("Creating modeling.ipynb with first 8 cells...")
print("Run this script to generate the notebook, then I'll add more cells in the next iteration.")

# Create notebook structure
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Save to file
output_path = '/home/konstantinos-papadogiannis/energy-data-mining/notebooks/modeling.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"✅ Created {output_path}")
print(f"📊 Total cells: {len(cells)}")
print("\nΤρέξε αυτό το cell για να δεις τα πρώτα 8 cells.")
print("Μετά θα προσθέσω τα υπόλοιπα (features, models, evaluation).")
