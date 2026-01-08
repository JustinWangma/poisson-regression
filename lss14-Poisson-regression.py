import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf


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

def run_analysis(df):

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
    return res

def plot_err_curve(df, res):
    # # 1. Load Data
    # data_path = r"data\lss14.csv"
    # try:
    #     df = pd.read_csv(data_path)
    # except FileNotFoundError:
    #     print("Error: lss14.csv not found.")
    #     return

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
    linear_beta =  round(res.x[5], 4)
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

def read_csv():
    # 1. Load Data
    data_path = r"data\lss14.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Error: lss14.csv not found.")
        return
    return df

if __name__ == "__main__":
    df=read_csv()
    res=run_analysis(df)
    plot_err_curve(df, res)