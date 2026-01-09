import pandas as pd
import numpy as np
import logging
import os
import warnings

from dowhy import CausalModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LassoCV


# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# 1. GLOBAL SEED (Locks down Numpy, which DoWhy uses for shuffling)
np.random.seed(42)

def run_causal_ml_analysis():
    # 1. Load Data
    df = load_and_prep_data()
    if df is None:
        return

    # Define Confounders (The background factors)
    confounders = ['age', 'agex', 'sex', 'city'] 
    
    print(f"Loaded Data: {len(df)} rows.")
    print("Outcome: Mortality Rate (per 10,000 Person-Years)")
    print("Treatment: Colon Dose (Gy)")
    print(f"Confounders: {confounders}\n")

    # 3. Setup the Causal Model (DoWhy)
    # We define the assumptions: Confounders affect both Dose and Mortality.
    model = CausalModel(
        data=df,
        treatment='dose_gy',
        outcome='mortality_rate_10k',
        common_causes=confounders
    )
    
    # Identify the estimand (The target we want to calculate)
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    # 4. Estimate Effect using Double Machine Learning (EconML)
    # Why LinearDML?
    # - It assumes the Treatment Effect is Linear (matches the paper's finding).
    # - But it allows the Background Rate (Age/Sex effects) to be non-linear 
    #   and complex, modeled by Random Forests.
    
    print("DoWhy-CMLTraining Causal ML Model (LinearDML with Random Forests)...")
    print("Note: This uses person-years as sample weights for accuracy.")
    
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.econml.dml.LinearDML",
        target_units="ate",
        method_params={
            "init_params": {
                "model_y": RandomForestRegressor(n_estimators=100, min_samples_leaf=20, random_state=42), # Models Risk ~ Age+Sex
                "model_t": RandomForestRegressor(n_estimators=100, min_samples_leaf=20, random_state=42), # Models Dose ~ Age+Sex
                # Fix 2: Lock the LinearDML Cross-Validation splitting
                "random_state": 42,
                "linear_first_stages": False,
                "discrete_treatment": False
            },
            "fit_params": {
                # CRITICAL: We weight the data by person-years.
                # A row with 10,000 PY is more important than a row with 1 PY.
                "sample_weight": df['pyr'].values
            }
        }
    )

    print("\n" + "="*40)
    print("DoWhy-CML RESULTS")
    print("="*40)
    print(f"Estimated ATE: {estimate.value:.4f}")
    print("(Interpretation: Excess deaths per 10,000 person-years for every 1 Gy of dose)")

    # 5. Robustness Check (Refutation)
    # Placebo Test: Replace the true dose with a random variable.
    # The effect should drop to roughly 0.
    print("\nDoWhy-CML Running Placebo Refutation (Sanity Check)...")
    refute = model.refute_estimate(
        identified_estimand, 
        estimate,
        method_name="placebo_treatment_refuter",
        placebo_type="permute",
        num_simulations=5 ,# Low number for speed; increase for rigorous check
        random_state=42  # Fix 3: Lock the Placebo Shuffling
    )
    print(refute)


# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def load_and_prep_data(filepath=r"data\lss14.csv"):
    """
    Loads the CSV, cleans it, and creates necessary columns.
    Returns a prepared DataFrame.
    """
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None

    df = pd.read_csv(filepath)
    
    # 1. Filter out rows with 0 person-years
    df = df[df['pyr'] > 0].copy()

    # 2. Create Outcome: Mortality Rate (per 10,000 PY)
    df['mortality_rate_10k'] = (df['solid'] / df['pyr']) * 10000

    # 3. Create Treatment: Dose in Gy (convert from mGy)
    df['dose_gy'] = df['colon10'] / 1000.0
    
    return df

def run_subgroup_analysis(min_age, max_age):
    """
    Runs the Causal ML analysis on a specific age range of survivors.
    """
    print("\n" + "="*60)
    print(f"DoWhy-CML SUBGROUP ANALYSIS: AGE {min_age} - {max_age}")
    print("="*60)

    # 1. Load Data
    df = load_and_prep_data()
    if df is None:
        return

    # 2. Filter for the specific Age Window
    original_count = len(df)
    subgroup_df = df[(df['age'] >= min_age) & (df['age'] <= max_age)].copy()
    
    print(f"Original Dataset: {original_count} rows")
    print(f"Filtered Dataset: {len(subgroup_df)} rows (Survivors aged {min_age}-{max_age})")

    # Check if we have enough data to run
    if len(subgroup_df) < 100:
        print("Error: Not enough data points in this age range to run ML model.")
        return

    # 3. Define Confounders
    # We still control for specific age/sex/city within this window
    confounders = ['age', 'agex', 'sex', 'city'] 

    # 4. Setup Causal Model (DoWhy)
    model = CausalModel(
        data=subgroup_df,
        treatment='dose_gy',
        outcome='mortality_rate_10k',
        common_causes=confounders
    )
    
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    # 5. Estimate Effect (LinearDML)
    print("Training Causal ML Model (LinearDML)...")
    
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.econml.dml.LinearDML",
        target_units="ate",
        method_params={
            "init_params": {
                # Using Random State 42 for reproducible results
                "model_y": RandomForestRegressor(n_estimators=100, min_samples_leaf=20, random_state=42), 
                "model_t": RandomForestRegressor(n_estimators=100, min_samples_leaf=20, random_state=42),
                "random_state": 42,  # Locks the Cross-Validation Splitter
                "linear_first_stages": False,
                "discrete_treatment": False
            },
            "fit_params": {
                "sample_weight": subgroup_df['pyr'].values
            }
        }
    )

    # 6. Output Results
    print("-" * 40)
    print(f"DoWhy-CML Estimated ATE (Age {min_age}-{max_age}): {estimate.value:.4f}")
    print("(Excess deaths per 10,000 person-years per Gy)")
    print("-" * 40)


if __name__ == "__main__":

    # run_causal_ml_analysis()
    run_subgroup_analysis(60, 80)
