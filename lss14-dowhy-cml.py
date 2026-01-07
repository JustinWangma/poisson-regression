import pandas as pd
import numpy as np
import logging
import os
import warnings
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

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

def plot_err_curve():
    # 1. Load Data
    data_path = r"data\lss14.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Error: lss14.csv not found.")
        return

    # Filter standard data
    df = df[df['pyr'] > 0].copy()
    
    # 2. Prepare Variables
    # Convert mGy to Gy
    df['dose_gy'] = df['colon10'] / 1000.0
    # Log variables for offset and age (Power model match)
    df['log_pyr'] = np.log(df['pyr'])
    df['log_age_70'] = np.log(df['age'] / 70.0)

    # 3. Create Dose Categories (Binned Analysis)
    # We create bins to calculate independent risk points
    # Bins: 0, 0.1, 0.2, 0.5, 1.0, 2.0, Max
    bins = [-0.001, 0.005, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    labels = range(len(bins)-1)
    df['dose_cat'] = pd.cut(df['dose_gy'], bins=bins, labels=labels)
    
    # Calculate the mean dose for each category (for plotting position)
    dose_means = df.groupby('dose_cat')['dose_gy'].mean()

    print("Fitting Categorical Poisson Model (for plotting points)...")
    
    # 4. Fit Categorical Poisson Model (Statsmodels)
    # We use 'C(dose_cat)' to treat each bin as a separate yes/no variable
    # Baseline: The first bin (0 dose) is automatically the reference (RR=1)
    formula = "solid ~ C(dose_cat) + C(sex) + C(city) + log_age_70"
    
    model = smf.glm(formula=formula, data=df, 
                    offset=df['log_pyr'], 
                    family=sm.families.Poisson()).fit()

    # 5. Extract Results for Plotting
    # Get coefficients and confidence intervals
    params = model.params
    conf_int = model.conf_int()
    
    # Lists to store plotting data
    x_vals = []
    y_vals = [] # Relative Risks
    y_err_lower = []
    y_err_upper = []

    # Loop through categories to build RR points
    # Note: The first category (0 dose) is the baseline, so RR=1, Dose=Mean of Bin 0
    x_vals.append(dose_means[0])
    y_vals.append(1.0)
    y_err_lower.append(0.0) 
    y_err_upper.append(0.0)

    for i in range(1, len(bins)-1):
        # Statsmodels naming convention: C(dose_cat)[T.i]
        col_name = f"C(dose_cat)[T.{i}]"
        
        if col_name in params:
            coef = params[col_name]
            ci_low = conf_int.loc[col_name, 0]
            ci_high = conf_int.loc[col_name, 1]
            
            # Convert Log-Rate to Rate Ratio (RR = exp(coef))
            rr = np.exp(coef)
            rr_low = np.exp(ci_low)
            rr_high = np.exp(ci_high)
            
            x_vals.append(dose_means[i])
            y_vals.append(rr)
            # Matplotlib error bars are relative lengths, not absolute positions
            y_err_lower.append(rr - rr_low)
            y_err_upper.append(rr_high - rr)

    # 6. Plotting
    plt.figure(figsize=(10, 6))

    # A. Plot the Data Points (Categorical Estimates)
    plt.errorbar(x_vals, y_vals, yerr=[y_err_lower, y_err_upper], 
                 fmt='o', color='black', capsize=5, label='Observed Risk (Categorical Fit)')

    # B. Plot the Linear Model (Step 1 Result)
    # Formula: ERR = 0.48 * Dose  -->  RR = 1 + 0.48 * Dose
    linear_beta = 0.48 
    x_line = np.linspace(0, 3.0, 100)
    y_line = 1 + linear_beta * x_line
    
    plt.plot(x_line, y_line, color='red', linewidth=2, 
             label=f'Linear Model (ERR = {linear_beta}/Gy)')

    # Formatting
    plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Radiation Dose (Gy)', fontsize=12)
    plt.ylabel('Relative Risk (RR)', fontsize=12)
    plt.title('Dose-Response Curve: Atomic Bomb Survivors\n(Solid Cancer Mortality)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 2.5) # Focus on the main data range
    
    # Save
    plt.savefig('lss_dose_response_curve.png', dpi=300)
    print("Plot saved as 'lss_dose_response_curve.png'")
    plt.show()

if __name__ == "__main__":
    # run_causal_ml_analysis()
    plot_err_curve()