import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm
from copy import deepcopy
import warnings
import os

# ==========================================
# 1. HELPER FUNCTIONS (Models & Tests)
# ==========================================

def background_rate(params, data):
    # RESTORED: Now includes 'b_sex'
    # Log-linear background: exp(Intercept + b_age*Age + b_year*Year + b_sex*Sex)
    intercept, b_age, b_year, b_sex = params[0], params[1], params[2], params[3]
    
    # Scale variables for stability
    # Sex is binary (0/1), so no scaling needed
    return np.exp(intercept + b_age * (data['Age_at_termination']/100.0) + 
                  b_year * ((data['Year_of_hire']-1900)/100.0) + 
                  b_sex * data['sex'])

def neg_log_likelihood_linear(params, data):
    # Model: Rate = Background * (1 + Beta * Dose_Gy)
    bkg = background_rate(params, data)
    err_beta = params[4] # Beta is now at index 4
    
    dose_gy = data['treatment'] / 1000.0
    
    rr = 1 + err_beta * dose_gy
    rr = np.maximum(rr, 1e-9) 
    
    # Time = Person-Years
    lambda_pred = bkg * rr * data['time']
    
    # Event = Deaths
    nll = np.sum(lambda_pred - data['event'] * np.log(lambda_pred))
    return nll

def neg_log_likelihood_lq(params, data):
    # Model: Rate = Background * (1 + Beta*Dose + Gamma*Dose^2)
    bkg = background_rate(params, data)
    err_beta = params[4]
    err_gamma = params[5] # Gamma is now at index 5
    
    dose_gy = data['treatment'] / 1000.0
    
    rr = 1 + err_beta * dose_gy + err_gamma * (dose_gy**2)
    rr = np.maximum(rr, 1e-9)
    
    lambda_pred = bkg * rr * data['time']
    
    nll = np.sum(lambda_pred - data['event'] * np.log(lambda_pred))
    return nll

def calculate_wald_test(model_func, result, data):
    """Calculates Standard Errors and P-values."""
    params = result.x
    n_params = len(params)
    hessian = np.zeros((n_params, n_params))
    epsilon = 1e-4 
    
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
        return [np.nan]*n_params, [np.nan]*n_params

# ==========================================
# 2. SIMULATION LOOP (With Sex Simulation)
# ==========================================

def run_simulation_traditional():
    # 1. Load Data
    data_path = r"data\sampled_data_rounded.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        return

    fullSet_sampled_base = pd.read_csv(data_path)
    print(f"Loaded data from {data_path} with {len(fullSet_sampled_base)} rows.")

    fake_effect_vals = [-0.04, 0, 0.04]
    censor_intercepts = [-1, -0.5] 
    
    results = []

    print(f"\n{'Sim Effect':<12} | {'Cens Int':<10} | {'Est Beta (ERR/Gy)':<20} | {'P-Value':<10} | {'LRT P-Val':<10}")
    print("-" * 75)

    for feff in fake_effect_vals:
        for cint in censor_intercepts:
            
            # --- A. Apply Simulation Logic ---
            df = deepcopy(fullSet_sampled_base)
            
            # 1. SIMULATE SEX (New Step)
            # Assign Sex: 0=Male (80%), 1=Female (20%)
            np.random.seed(42) # Ensure consistency across loops
            df['sex'] = np.random.binomial(1, 0.2, size=len(df))
            
            # 2. Simulate Survival Time
            # Note: We incorporate 'sex' into the "Truth" logic to make it realistic.
            # Females typically live longer, so we add a small boost (e.g., +2 years)
            baseline_val = 30.
            
            df["time"] = np.maximum(0.01, (
                baseline_val / (1 + 5.*(df["Age_at_termination"]/100.)**4
                                + 2.5*((df["Year_of_hire"]-1900.)/100.)**2
                                - 0.2 * df['sex']) # Females (1) get smaller divisor -> larger time
                + feff * df["treatment"] 
                + np.random.normal(0, 0.05, size=df.shape[0])
            ))
            
            # 3. Simulate Censoring
            logits = cint + 3. * df["Age_at_termination"]/100.
            censor_probs = np.clip(np.exp(logits) / (1+np.exp(logits)), 0.01, 0.99)
            is_event = np.random.binomial(1, 1 - censor_probs)
            
            df["event"] = is_event
            censored_prop = 1 - df["event"].mean()

            # --- B. Fit Traditional Models ---
            
            # Linear Model (Estimating Beta)
            # Initial guess: Intercept=0, Age=5, Year=2.5, Sex=-0.5, Beta=0
            # Added 'Sex' parameter (index 3)
            init_linear = [0, 5, 2.5, -0.5, 0.0] 
            bounds_linear = [(None,None), (None,None), (None,None), (None,None), (-5, 50)]
            
            res_linear = minimize(
                neg_log_likelihood_linear, init_linear, args=(df), 
                method='L-BFGS-B', bounds=bounds_linear
            )
            
            # LQ Model (Estimating Gamma)
            init_lq = list(res_linear.x) + [0.0]
            bounds_lq = bounds_linear + [(-10, 10)]
            
            res_lq = minimize(
                neg_log_likelihood_lq, init_lq, args=(df), 
                method='L-BFGS-B', bounds=bounds_lq
            )
            
            # --- C. Calculate Statistics ---
            se, p_vals = calculate_wald_test(neg_log_likelihood_linear, res_linear, df)
            
            # Results
            est_beta = res_linear.x[4] # Beta is now index 4
            beta_p_val = p_vals[4]
            d_nll = 2 * (res_linear.fun - res_lq.fun)
            lrt_p_val = chi2.sf(d_nll, df=1)
            
            print(f"{feff:<12} | {cint:<10} | {est_beta:<20.4f} | {beta_p_val:<10.4f} | {lrt_p_val:<10.4f}")
            
            results.append({
                "fake_effect_val": feff,
                "censor_intercept": cint,
                "censored_prop": censored_prop,
                "est_beta_err_gy": est_beta,
                "beta_p_value": beta_p_val,
                "lrt_curvature_p_value": lrt_p_val,
                "convergence_success": res_linear.success
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv("simulation_results_traditional.csv", index=False)
    print("\n✅ Simulation complete. Results saved to 'simulation_results_traditional.csv'")

if __name__ == "__main__":
    run_simulation_traditional()