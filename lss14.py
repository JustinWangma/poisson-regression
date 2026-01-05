import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm
import os

# ==========================================
# 1. HELPER FUNCTIONS (Poisson Model)
# ==========================================

def background_rate(params, data):
    """
    Background rate model for aggregated data.
    Params: 
      0: Intercept
      1: Coeff for Age (Attained Age)
      2: Coeff for AgeX (Age at exposure)
      3: Coeff for Sex (1=Male, 2=Female)
      4: Coeff for City (1=Hiroshima, 2=Nagasaki)
    """
    intercept = params[0]
    b_age = params[1]
    b_agex = params[2]
    b_sex = params[3]
    b_city = params[4]
    
    # Scale continuous variables for numerical stability (e.g., divide by 100)
    # Using 'sex' directly (1 or 2) is fine, or you can map to 0/1. 
    # Here we treat sex/city as continuous or indicator-like for simplicity.
    
    log_rate = (intercept + 
                b_age * np.log(data['age']/70.0) + # Centered at 70 years
                b_agex * (data['agex']/100.0) + 
                b_sex * (data['sex'] == 1).astype(int) +  # 1 if Male
                b_city * (data['city'] == 1).astype(int)) # 1 if Hiroshima
                
    return np.exp(log_rate)


def neg_log_likelihood_linear(params, data):
    # Model: Expected Deaths = Background * (1 + Beta * Dose) * PYR
    # params structure: [Intercept, Age, AgeX, Sex, City, Beta_Dose]
    
    # 1. Calculate Background Rate
    bkg = background_rate(params[:5], data)
    
    # 2. Calculate Excess Relative Risk (ERR)
    err_beta = params[5] 
    dose_gy = data['colon10'] / 1000.0 # colon10 is in mGy, convert to Gy
    
    rr = 1 + err_beta * dose_gy
    # Ensure RR is non-negative (model constraint)
    rr = np.maximum(rr, 1e-9) 
    
    # 3. Predicted Counts (Lambda)
    # Expected Deaths = Rate * RR * PersonYears
    lambda_pred = bkg * rr * data['pyr']
    
    # 4. Poisson Negative Log Likelihood
    # Formula: sum(lambda - k * ln(lambda))  (ignoring log(k!) term as it's constant)
    # data['solid'] is the observed count of cancer deaths
    nll = np.sum(lambda_pred - data['solid'] * np.log(lambda_pred))
    
    return nll

def calculate_wald_test(model_func, result, data):
    """Calculates Standard Errors and P-values via Hessian Inversion."""
    params = result.x
    n_params = len(params)
    hessian = np.zeros((n_params, n_params))
    epsilon = 1e-4 
    
    # Compute Hessian matrix (2nd derivatives)
    for i in range(n_params):
        for j in range(n_params):
            p_ij = params.copy()
            p_ij[i] += epsilon; p_ij[j] += epsilon; f_pp = model_func(p_ij, data)
            p_ij = params.copy(); p_ij[i] -= epsilon; p_ij[j] -= epsilon; f_mm = model_func(p_ij, data)
            p_ij = params.copy(); p_ij[i] += epsilon; p_ij[j] -= epsilon; f_pm = model_func(p_ij, data)
            p_ij = params.copy(); p_ij[i] -= epsilon; p_ij[j] += epsilon; f_mp = model_func(p_ij, data)
            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
            
    try:
        cov_matrix = np.linalg.inv(hessian)
        std_errors = np.sqrt(np.diag(cov_matrix))
        z_scores = params / std_errors
        p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
        return std_errors, p_values
    except:
        print("Matrix inversion failed (Hessian singular).")
        return [np.nan]*n_params, [np.nan]*n_params

# ==========================================
# 2. MAIN ANALYSIS
# ==========================================

def run_analysis():
    # 1. Load Real Data
    data_path = r"data\lss14.csv"
    # Ensure lss14.csv is in the same folder or update path
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"Loaded lss14.csv with {len(df)} rows.")
    else:
        print("Error: lss14.csv not found.")
        return

    # 2. Data Cleaning
    # Remove rows with 0 person-years (cannot model rate)
    df = df[df['pyr'] > 0].copy()
    
    # 3. Fit Model
    # Initial Guess: Intercept, Age, AgeX, Sex, City, Beta
    init_params = [-3.0, 5.0, 0.5, 0.2, 0.0, 0.5] 
    
    print("\nFitting Poisson Regression Model...")
    res = minimize(
        neg_log_likelihood_linear, 
        init_params, 
        args=(df), 
        method='L-BFGS-B', 
        # Bounds: None for bkg params, Dose beta > -0.5 to prevent RR<0 issues
        bounds=[(None,None)]*5 + [(-0.9, 10)] 
    )

    # 4. Output Results
    if res.success:
        print("\nOptimization Successful!")
        se, p_vals = calculate_wald_test(neg_log_likelihood_linear, res, df)
        
        param_names = ['Intercept', 'Attained Age', 'Age at Exposure', 'Sex (Male)', 'City (Hiroshima)', 'ERR/Gy (Dose)']
        
        print(f"\n{'Parameter':<20} | {'Estimate':<10} | {'Std Error':<10} | {'P-Value':<10}")
        print("-" * 60)
        for i, name in enumerate(param_names):
            print(f"{name:<20} | {res.x[i]:<10.4f} | {se[i]:<10.4f} | {p_vals[i]:<10.4e}")
            
        print(f"\nExcess Relative Risk (ERR) per Gy: {res.x[5]:.4f}")
    else:
        print("Optimization Failed:", res.message)

if __name__ == "__main__":
    run_analysis()