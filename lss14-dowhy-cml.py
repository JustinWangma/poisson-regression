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

def run_causal_ml_analysis():
    # 1. Load Data
    data_path = r"data\lss14.csv"
    if not os.path.exists(data_path):
        print("Error: lss14.csv not found.")
        return

    df = pd.read_csv(data_path)
    
    # 2. Data Preparation
    # Filter out rows with 0 person-years to avoid division errors
    df = df[df['pyr'] > 0].copy()

    # Create the Outcome: Mortality Rate (Deaths per Person-Year)
    # We multiply by 10,000 to make the numbers readable (Deaths per 10k PY)
    df['mortality_rate_10k'] = (df['solid'] / df['pyr']) * 10000

    # Create Treatment: Dose in Gy (convert from mGy)
    df['dose_gy'] = df['colon10'] / 1000.0

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
    
    print("Training Causal ML Model (LinearDML with Random Forests)...")
    print("Note: This uses person-years as sample weights for accuracy.")
    
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.econml.dml.LinearDML",
        target_units="ate",
        method_params={
            "init_params": {
                "model_y": RandomForestRegressor(n_estimators=100, min_samples_leaf=10), # Models Risk ~ Age+Sex
                "model_t": RandomForestRegressor(n_estimators=100, min_samples_leaf=10), # Models Dose ~ Age+Sex
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
    print("CAUSAL ML RESULTS")
    print("="*40)
    print(f"Estimated ATE: {estimate.value:.4f}")
    print("(Interpretation: Excess deaths per 10,000 person-years for every 1 Gy of dose)")

    # 5. Robustness Check (Refutation)
    # Placebo Test: Replace the true dose with a random variable.
    # The effect should drop to roughly 0.
    print("\nRunning Placebo Refutation (Sanity Check)...")
    refute = model.refute_estimate(
        identified_estimand, 
        estimate,
        method_name="placebo_treatment_refuter",
        placebo_type="permute",
        num_simulations=5 # Low number for speed; increase for rigorous check
    )
    print(refute)



if __name__ == "__main__":
    run_causal_ml_analysis()
