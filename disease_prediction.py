# ==============================================================================
# FINAL SCRIPT FOR THE (920, 16) DATASET
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

try:
    # --- 1. LOAD DATA ---
    # This assumes your file is named 'heart_disease_uci.csv'
    df = pd.read_csv('heart_disease_uci.csv')
    print("Step 1: File loaded successfully.")

    # --- 2. CLEAN DATA ---
    # This dataset uses '?' for missing data. We replace it and make everything a number.
    df = df.replace('?', np.nan)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    print("Step 2: Data cleaned successfully.")

    # --- 3. PREPARE DATA ---
    # The target column in this dataset is the last one.
    # We will select it automatically.
    target_column = df.columns[-1]
    print(f"Identified target column: '{target_column}'")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Make the target binary (0 for no disease, 1 for disease)
    y.loc[y > 0] = 1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Step 3: Data prepared for model.")

    # --- 4. TRAIN AND EVALUATE ---
    model = LogisticRegression(max_iter=2000) # Increased max_iter for this complex data
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("\n--- FINAL RESULTS ---")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

except Exception as e:
    print("\n--- AN ERROR OCCURRED ---")
    print(f"Error: {e}")
    print("\nPlease double-check that the file 'heart_disease_uci.csv' is uploaded correctly.")