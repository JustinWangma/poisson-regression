import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor
from econml.dml import LinearDML

# Suppress warnings
warnings.filterwarnings("ignore")

def run_econml_direct(min_age=60, max_age=80):
    print("="*60)
    print(f"ECON-ML DIRECT (Subgroup: Age {min_age}-{max_age})")
    print("="*60)

    # 1. Load & Prep Data (Same as before)
    data_path = r"data\lss14.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Error: lss14.csv not found.")
        return

    # Filter bad rows and apply Subgroup
    df = df[df['pyr'] > 0].copy()
    df = df[(df['age'] >= min_age) & (df['age'] <= max_age)].copy()
    
    # 2. Define Arrays (The "Raw" Math)
    # Outcome (Y)
    Y = (df['solid'] / df['pyr']) * 10000 
    
    # Treatment (T)
    T = df['colon10'] / 1000.0
    
    # Controls (W) - The Confounders
    # We treat these as "Nuisance Parameters" to be filtered out
    W = df[['age', 'agex', 'sex', 'city']]
    
    # Sample Weights (Critical for accuracy)
    weights = df['pyr'].values

    print(f"Data Size: {len(df)} rows")
    print("Training LinearDML (EconML Direct)...")

    # 3. Configure the Estimator
    # We use the exact same Random Forest settings as the DoWhy script
    est = LinearDML(
        model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=20, random_state=42),
        model_t=RandomForestRegressor(n_estimators=100, min_samples_leaf=20, random_state=42),
        linear_first_stages=False,
        discrete_treatment=False,
        random_state=42
    )

    # 4. Fit the Model
    # Notice how we pass 'W' (controls) and 'sample_weight' directly
    est.fit(Y, T, W=W, sample_weight=weights)

    # 5. Calculate Average Treatment Effect (ATE)
    ate = est.ate() # Calculates the coefficient
    
    print("-" * 40)
    print(f"ECON-ML Estimated ATE: {ate:.4f}")
    print("-" * 40)
    
    return ate

if __name__ == "__main__":
    # Run the "Gold Standard" Age 60-80 check
    run_econml_direct(60, 80)