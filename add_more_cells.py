#!/usr/bin/env python3
"""
Script to add more cells to modeling.ipynb
Adds: Feature selection + Logistic Regression + Random Forest + XGBoost
"""

import json

# Load existing notebook
notebook_path = '/home/konstantinos-papadogiannis/energy-data-mining/notebooks/modeling.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# New cells to add
new_cells = [
    # Cell 8: Feature selection header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## 4. Επιλογή Features (Χωρίς Data Leakage)\n",
            "\n",
            "### Κανόνας\n",
            "Χρησιμοποιούμε **μόνο** features που είναι γνωστά **πριν** την ημέρα που θέλουμε να προβλέψουμε.\n",
            "\n",
            "### ✅ Safe Features (OK to use)\n",
            "1. **LAG features**: `lag_1`, `lag_7`, `lag_14`, `lag_30` (χθεσινή, προηγούμενης εβδομάδας κτλ)\n",
            "2. **Rolling statistics**: `rolling_mean_7d`, `rolling_std_7d`, κτλ (υπολογισμένα από παρελθόν)\n",
            "3. **EMA**: `ema_7d`, `ema_30d` (exponential moving average)\n",
            "4. **Differences**: `diff_1d`, `diff_7d` (αλλαγές από χθες/προηγούμενη εβδομάδα)\n",
            "5. **Calendar features**: `DayOfWeek`, `Month`, `Season`, `IsWeekend`, κτλ (γνωστά εκ των προτέρων)\n",
            "\n",
            "### ❌ Forbidden Features (Data Leakage!)\n",
            "- `Daily_total_power` (αυτό προσπαθούμε να προβλέψουμε!)\n",
            "- `Daily_mean_power`, `Daily_peak_power` (same-day aggregates)\n",
            "- `Peak_hour_power`, `Nighttime_usage` (same-day)\n",
            "- `Total_submetering`, `Sub1_ratio`, κτλ (same-day)\n",
            "- `Next_day_consumption` (μελλοντική τιμή!)"
        ]
    },
    
    # Cell 9: Define safe features
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Ορισμός safe features\n",
            "safe_features = [\n",
            "    # LAG features (past values)\n",
            "    'lag_1', 'lag_7', 'lag_14', 'lag_30',\n",
            "    \n",
            "    # Rolling statistics (computed from past)\n",
            "    'rolling_mean_7d', 'rolling_std_7d', 'rolling_median_7d',\n",
            "    'rolling_min_7d', 'rolling_max_7d',\n",
            "    \n",
            "    # Exponential moving averages\n",
            "    'ema_7d', 'ema_30d',\n",
            "    \n",
            "    # Differences (changes from past)\n",
            "    'diff_1d', 'diff_7d',\n",
            "    \n",
            "    # Calendar features (known in advance)\n",
            "    'DayOfWeek', 'IsWeekend', 'Month', 'Season', 'Year', 'DayOfYear'\n",
            "]\n",
            "\n",
            "# Έλεγχος ότι όλα τα features υπάρχουν\n",
            "missing_features = [f for f in safe_features if f not in train.columns]\n",
            "if missing_features:\n",
            "    print(f'⚠️  Missing features: {missing_features}')\n",
            "    safe_features = [f for f in safe_features if f in train.columns]\n",
            "\n",
            "print(f'✅ Using {len(safe_features)} safe features (no data leakage):')\n",
            "for i, f in enumerate(safe_features, 1):\n",
            "    print(f'  {i:2d}. {f}')"
        ]
    },
    
    # Cell 10: Create X and y
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Δημιουργία X (features) και y (target) με ευθυγράμμιση\n",
            "X_train = train[safe_features].copy()\n",
            "X_val = val[safe_features].copy()\n",
            "X_test = test[safe_features].copy()\n",
            "\n",
            "# Αφαίρεση NaN rows (από lag/rolling features)\n",
            "train_mask = X_train.notna().all(axis=1)\n",
            "X_train = X_train[train_mask]\n",
            "y_train_aligned = y_train[train_mask]\n",
            "\n",
            "val_mask = X_val.notna().all(axis=1)\n",
            "X_val = X_val[val_mask]\n",
            "y_val_aligned = y_val[val_mask]\n",
            "\n",
            "test_mask = X_test.notna().all(axis=1)\n",
            "X_test = X_test[test_mask]\n",
            "y_test_aligned = y_test[test_mask]\n",
            "\n",
            "print('📊 Final Dataset Shapes (after removing NaN):')\n",
            "print(f'  X_train: {X_train.shape}, y_train: {y_train_aligned.shape}')\n",
            "print(f'  X_val:   {X_val.shape}, y_val:   {y_val_aligned.shape}')\n",
            "print(f'  X_test:  {X_test.shape}, y_test:  {y_test_aligned.shape}')\n",
            "\n",
            "print('\\n✅ No missing values in features')\n",
            "print(f'✅ X and y are aligned (same indices)')"
        ]
    },
    
    # Cell 11: Preprocessing pipeline header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## 5. Preprocessing Pipeline\n",
            "\n",
            "Δημιουργούμε pipeline που:\n",
            "1. **OneHotEncoder** για categorical features (`Season`)\n",
            "2. **StandardScaler** για numeric features\n",
            "3. Fit **μόνο στο train**, transform σε train/val/test"
        ]
    },
    
    # Cell 12: Create preprocessor
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Διαχωρισμός σε categorical και numeric features\n",
            "categorical_features = ['Season']\n",
            "numeric_features = [f for f in safe_features if f not in categorical_features]\n",
            "\n",
            "print(f'📊 Feature Types:')\n",
            "print(f'  Categorical: {categorical_features}')\n",
            "print(f'  Numeric: {len(numeric_features)} features')\n",
            "\n",
            "# Preprocessing pipeline\n",
            "preprocessor = ColumnTransformer(\n",
            "    transformers=[\n",
            "        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),\n",
            "        ('num', StandardScaler(), numeric_features)\n",
            "    ],\n",
            "    remainder='drop'\n",
            ")\n",
            "\n",
            "print('\\n✅ Preprocessing pipeline created')\n",
            "print('   - OneHotEncoder for Season (4 categories → 4 binary columns)')\n",
            "print('   - StandardScaler for numeric features (mean=0, std=1)')"
        ]
    },
    
    # Cell 13: Helper function
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Helper function για αξιολόγηση μοντέλων\n",
            "def evaluate_model(y_true, y_pred, y_proba, set_name=''):\n",
            "    \"\"\"Υπολογίζει και εκτυπώνει μετρικές αξιολόγησης\"\"\"\n",
            "    acc = accuracy_score(y_true, y_pred)\n",
            "    prec = precision_score(y_true, y_pred, zero_division=0)\n",
            "    rec = recall_score(y_true, y_pred, zero_division=0)\n",
            "    f1 = f1_score(y_true, y_pred, zero_division=0)\n",
            "    auc = roc_auc_score(y_true, y_proba)\n",
            "    \n",
            "    print(f'\\n📊 {set_name} Set Results:')\n",
            "    print(f'  Accuracy:  {acc*100:.2f}%')\n",
            "    print(f'  Precision: {prec*100:.2f}%')\n",
            "    print(f'  Recall:    {rec*100:.2f}%')\n",
            "    print(f'  F1-Score:  {f1*100:.2f}%')\n",
            "    print(f'  ROC-AUC:   {auc*100:.2f}%')\n",
            "    \n",
            "    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}\n",
            "\n",
            "print('✅ Helper function defined')"
        ]
    },
    
    # Cell 14: Logistic Regression header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "---\n",
            "# ΜΟΝΤΕΛΟ 1: Logistic Regression (Baseline)\n",
            "\n",
            "## Γιατί Logistic Regression;\n",
            "- ✅ **Γρήγορο** και απλό\n",
            "- ✅ **Interpretable** (μπορούμε να δούμε τα coefficients)\n",
            "- ✅ Καλό **baseline** για σύγκριση με πιο πολύπλοκα μοντέλα\n",
            "- ✅ Λειτουργεί καλά όταν τα features είναι κανονικοποιημένα\n",
            "\n",
            "## Παράμετροι\n",
            "- `max_iter=1000`: Αρκετές επαναλήψεις για σύγκλιση\n",
            "- `random_state=42`: Για reproducibility"
        ]
    },
    
    # Cell 15: Train Logistic Regression
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Δημιουργία pipeline: preprocessing + model\n",
            "logistic_pipeline = Pipeline([\n",
            "    ('preprocessor', preprocessor),\n",
            "    ('classifier', LogisticRegression(max_iter=1000, random_state=42))\n",
            "])\n",
            "\n",
            "print('🤖 Model: Logistic Regression')\n",
            "print('   Parameters: max_iter=1000, random_state=42')\n",
            "print('\\n🔄 Training...')\n",
            "\n",
            "# Εκπαίδευση\n",
            "logistic_pipeline.fit(X_train, y_train_aligned)\n",
            "\n",
            "print('✅ Training completed!')"
        ]
    },
    
    # Cell 16: Evaluate Logistic Regression
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Προβλέψεις\n",
            "y_train_pred_lr = logistic_pipeline.predict(X_train)\n",
            "y_val_pred_lr = logistic_pipeline.predict(X_val)\n",
            "y_test_pred_lr = logistic_pipeline.predict(X_test)\n",
            "\n",
            "# Probabilities (για ROC-AUC)\n",
            "y_train_proba_lr = logistic_pipeline.predict_proba(X_train)[:, 1]\n",
            "y_val_proba_lr = logistic_pipeline.predict_proba(X_val)[:, 1]\n",
            "y_test_proba_lr = logistic_pipeline.predict_proba(X_test)[:, 1]\n",
            "\n",
            "print('='*70)\n",
            "print('LOGISTIC REGRESSION - EVALUATION')\n",
            "print('='*70)\n",
            "\n",
            "lr_train_metrics = evaluate_model(y_train_aligned, y_train_pred_lr, y_train_proba_lr, 'Train')\n",
            "lr_val_metrics = evaluate_model(y_val_aligned, y_val_pred_lr, y_val_proba_lr, 'Validation')\n",
            "lr_test_metrics = evaluate_model(y_test_aligned, y_test_pred_lr, y_test_proba_lr, 'Test')\n",
            "\n",
            "print('\\n' + '='*70)"
        ]
    },
    
    # Cell 17: Classification Report
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print('📋 Detailed Classification Report (Validation Set):\\n')\n",
            "print(classification_report(y_val_aligned, y_val_pred_lr, \n",
            "                          target_names=['Normal/Low (0)', 'High (1)'],\n",
            "                          digits=3))"
        ]
    },
    
    # Cell 18: Confusion Matrix & ROC Curve
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Visualization\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# Confusion Matrix\n",
            "cm = confusion_matrix(y_val_aligned, y_val_pred_lr)\n",
            "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],\n",
            "            xticklabels=['Normal/Low', 'High'],\n",
            "            yticklabels=['Normal/Low', 'High'])\n",
            "axes[0].set_title('Confusion Matrix - Logistic Regression (Validation)', fontsize=13, fontweight='bold')\n",
            "axes[0].set_xlabel('Predicted Label')\n",
            "axes[0].set_ylabel('True Label')\n",
            "\n",
            "# ROC Curve\n",
            "RocCurveDisplay.from_predictions(y_val_aligned, y_val_proba_lr, ax=axes[1])\n",
            "axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')\n",
            "axes[1].set_title('ROC Curve - Logistic Regression (Validation)', fontsize=13, fontweight='bold')\n",
            "axes[1].legend()\n",
            "axes[1].grid(alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig(RESULTS_DIR / 'logistic_regression_eval.png', dpi=300, bbox_inches='tight')\n",
            "plt.show()\n",
            "\n",
            "print('✅ Visualization saved to results/logistic_regression_eval.png')"
        ]
    },
]

# Add new cells to notebook
notebook['cells'].extend(new_cells)

# Save updated notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"✅ Added {len(new_cells)} new cells to modeling.ipynb")
print(f"📊 Total cells now: {len(notebook['cells'])}")
print("\nΝέα cells (8-18):")
print("  - Cell 8-10: Feature Selection")
print("  - Cell 11-13: Preprocessing Pipeline & Helper Functions")
print("  - Cell 14-18: Logistic Regression (Training + Evaluation + Visualizations)")
print("\nΤρέξε τα νέα cells στο notebook!")
