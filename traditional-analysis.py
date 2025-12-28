import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm

# ==========================================
# 1. MOCK DATA GENERATION (Simulating Oak Ridge)
# ==========================================
def generate_mock_data(n=11299):
    """
    Generates synthetic data mimicking the Oak Ridge cohort structure.
    """
    np.random.seed(42) # Fixed seed for reproducibility
    
    # Covariates
    age_entry = np.random.normal(25, 5, n)
    follow_up_years = np.random.uniform(5, 40, n)
    attained_age = age_entry + follow_up_years
    sex = np.random.binomial(1, 0.75, n) # 75% Male
    
    # Dose: Lognormal (Most low, some high)
    dose_msv = np.random.lognormal(mean=2.5, sigma=1.2, size=n)
    dose_gy = dose_msv / 1000.0 # Convert to Gray
    
    # TRUE BIOLOGY (Hidden Truth):
    # Here we simulate a "Null" effect (Radiation Beta = 0) to see if the model behaves.
    # Change '0.0' to '0.5' below to simulate a harmful effect.
    true_beta = 0.0 
    
    baseline_risk = np.exp(-5 + 0.05 * attained_age + 0.3 * sex)
    radiation_multiplier = 1 + true_beta * dose_gy
    
    lambda_rate = baseline_risk * radiation_multiplier
    expected_deaths = lambda_rate * follow_up_years
    
    deaths = np.random.poisson(expected_deaths)
    
    df = pd.DataFrame({
        'person_years': follow_up_years,
        'attained_age': attained_age,
        'sex': sex,
        'dose_gy': dose_gy,
        'deaths': deaths
    })
    
    return df

# ==========================================
# 2. DEFINE THE MODELS (Grant/Preston Style)
# ==========================================

def background_rate(params, data):
    # Log-linear background: exp(Intercept + Age + Sex)
    intercept, b_age, b_sex = params[0], params[1], params[2]
    return np.exp(intercept + b_age * data['attained_age'] + b_sex * data['sex'])

def neg_log_likelihood_linear(params, data):
    # Model: Rate = Background * (1 + Beta * Dose)
    bkg = background_rate(params, data)
    err_beta = params[3]
    
    rr = 1 + err_beta * data['dose_gy']
    rr = np.maximum(rr, 1e-9) # Safety floor
    
    lambda_pred = bkg * rr * data['person_years']
    
    # Negative Log Likelihood
    nll = np.sum(lambda_pred - data['deaths'] * np.log(lambda_pred))
    return nll

def neg_log_likelihood_lq(params, data):
    # Model: Rate = Background * (1 + Beta*Dose + Gamma*Dose^2)
    bkg = background_rate(params, data)
    err_beta = params[3]
    err_gamma = params[4]
    
    rr = 1 + err_beta * data['dose_gy'] + err_gamma * (data['dose_gy']**2)
    rr = np.maximum(rr, 1e-9)
    
    lambda_pred = bkg * rr * data['person_years']
    
    nll = np.sum(lambda_pred - data['deaths'] * np.log(lambda_pred))
    return nll

# ==========================================
# 3. STATISTICAL TESTS (Wald & LRT)
# ==========================================

def calculate_wald_test(model_func, result, data):
    """
    Calculates Standard Errors and P-values using Numerical Hessian.
    Returns: (Standard Errors, P-values)
    """
    params = result.x
    n_params = len(params)
    hessian = np.zeros((n_params, n_params))
    epsilon = 1e-4 # Perturbation size
    
    # Approximate the Hessian (Curvature of the likelihood)
    for i in range(n_params):
        for j in range(n_params):
            p_ij = params.copy()
            
            # Central Difference Formula
            p_ij[i] += epsilon; p_ij[j] += epsilon; f_pp = model_func(p_ij, data)
            p_ij = params.copy(); p_ij[i] -= epsilon; p_ij[j] -= epsilon; f_mm = model_func(p_ij, data)
            p_ij = params.copy(); p_ij[i] += epsilon; p_ij[j] -= epsilon; f_pm = model_func(p_ij, data)
            p_ij = params.copy(); p_ij[i] -= epsilon; p_ij[j] += epsilon; f_mp = model_func(p_ij, data)
            
            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
            
    # Invert Hessian to get Variance-Covariance Matrix
    try:
        cov_matrix = np.linalg.inv(hessian)
        std_errors = np.sqrt(np.diag(cov_matrix))
        z_scores = params / std_errors
        p_values = 2 * (1 - norm.cdf(np.abs(z_scores))) # Two-tailed test
        return std_errors, p_values
    except:
        return None, None

# ==========================================
# 4. FIT AND COMPARE
# ==========================================

def fit_models(df):
    print(f"Analyzing {len(df)} workers with {df['deaths'].sum()} deaths...")
    
    # --- A. Fit Linear Model ---
    # Bounds: Intercept(Any), Age(Any), Sex(Any), Beta(-2 to 10)
    bounds_linear = [(None,None), (None,None), (None,None), (-2, 10)]
    init_linear = [-5, 0.05, 0.3, 0.1] 
    
    res_linear = minimize(
        neg_log_likelihood_linear, init_linear, args=(df), 
        method='L-BFGS-B', bounds=bounds_linear
    )
    
    # --- B. Fit Linear-Quadratic (LQ) Model ---
    # Bounds: Same + Gamma(-5 to 5)
    bounds_lq = bounds_linear + [(-5, 5)]
    init_lq = list(res_linear.x) + [0.01]
    
    res_lq = minimize(
        neg_log_likelihood_lq, init_lq, args=(df), 
        method='L-BFGS-B', bounds=bounds_lq
    )

    # --- C. Calculate Statistics ---
    se_lin, p_lin = calculate_wald_test(neg_log_likelihood_linear, res_linear, df)
    
    # Likelihood Ratio Test (LRT) for Curvature
    d_nll = 2 * (res_linear.fun - res_lq.fun)
    p_value_curvature = chi2.sf(d_nll, df=1)

    # ==========================================
    # 5. REPORT GENERATION
    # ==========================================
    print("\n" + "="*40)
    print("   COMPARISON REPORT: PARAMETRIC MODELS")
    print("="*40)
    
    # 1. Slope Analysis (Beta)
    beta = res_linear.x[3]
    beta_p = p_lin[3] if p_lin is not None else 1.0
    
    print(f"\n[TEST 1] LINEAR SLOPE (Risk Check)")
    print(f"  Estimated Excess Relative Risk (ERR/Gy): {beta:.4f}")
    print(f"  P-Value (Wald Test): {beta_p:.4f}")
    
    if beta_p < 0.05:
        print("  >> CONCLUSION: Significant Linear Risk Detected.")
        print("     (If Causal ML found NULL, this suggests Parametric Model is 'Hallucinating' risk)")
    else:
        print("  >> CONCLUSION: No Significant Risk Detected (Null Result).")
        print("     (Matches the Causal ML 'Null' finding)")

    # 2. Curvature Analysis (Gamma)
    gamma = res_lq.x[4]
    
    print(f"\n[TEST 2] CURVATURE (Shape Check)")
    print(f"  Estimated Curvature (Gamma): {gamma:.4f}")
    print(f"  LRT P-Value: {p_value_curvature:.4f}")
    
    if p_value_curvature < 0.05:
        print("  >> CONCLUSION: Significant Curvature Detected.")
        print("     (The Linear model is Misspecified. Use LQ model.)")
    else:
        print("  >> CONCLUSION: No Significant Curvature.")
        print("     (The Linear model is sufficient; no complex shape found.)")
        
    print("\n" + "="*40)

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Generate dummy data
    df_mock = generate_mock_data()
    
    # 2. Run the analysis
    fit_models(df_mock)