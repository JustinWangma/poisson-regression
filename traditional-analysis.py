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



# =========minimize function =================================
    # res_linear = minimize(
    #     neg_log_likelihood_linear, init_linear, args=(df), 
    #     method='L-BFGS-B', bounds=bounds_linear
    # )
# This block of code is the "cockpit" of the analysis. It serves as the command that launches the optimization process.

# Here is the breakdown of each component:

# ### 1. The Function: `neg_log_likelihood_linear`

# This is the **"Error Calculator."**

# * You are telling the solver: *"This is the function I want you to make as small as possible."*
# * Every time the solver makes a guess, it runs this function to see how "bad" that guess is (the Negative Log-Likelihood).

# ### 2. The Starting Point: `init_linear`

# This is the **"Initial Guess."**

# * Optimization algorithms are like hikers trying to find the bottom of a valley in the dark. They need a starting location.
# * `init_linear` is a list of 4 numbers (e.g., `[-5, 0.05, 0.3, 0.1]`).
# * If your guess is too wild, the solver might get lost. If it is reasonable, the solver will quickly find the true answer.

# ### 3. The Data: `args=(df)`

# This is the **"Fixed Information."**

# * Your likelihood function takes two inputs:
# 1. **Params:** The variables the solver *is allowed* to change (beta, gamma, etc.).
# 2. **Data:** The variables the solver *must not* change (the actual ages and doses of the workers).


# * `args=(df)` passes your dataframe into the function safely so the math can run, while keeping it locked so the solver doesn't modify the data itself.

# ### 4. The Algorithm: `method='L-BFGS-B'`

# This is the **"Search Strategy."**

# * **L-BFGS:** A smart mathematical algorithm ("Limited-memory Broyden–Fletcher–Goldfarb–Shanno") that estimates the slope of the hill to figure out which way is "down." It is much faster than guessing randomly.
# * **-B (Bound):** This version of the algorithm respects **Boundaries**. It ensures the solver never tries numbers that are forbidden (defined in the next step).

# ### 5. The Constraints: `bounds=[...]`

# These are the **"Guardrails."**
# You have 4 parameters in your model, so you provide a list of 4 rules:

# * `bounds=[(None,None), (None,None), (None,None), (-2, 10)]`

# | Parameter | Bound | Meaning |
# | --- | --- | --- |
# | **Intercept** | `(None, None)` | Can be any number (-∞ to +∞). |
# | **Age Effect** | `(None, None)` | Can be any number. |
# | **Sex Effect** | `(None, None)` | Can be any number. |
# | **Rad Beta** | `(-2, 10)` | **Constraint:** The radiation risk slope must be between -2 and 10. |

# * **Why constrain Beta?**
# * **Lower Bound (-2):** Mathematically, if Beta is too negative (e.g., -5), the formula `1 + beta*dose` becomes negative. You cannot take the log of a negative number; the code would crash. This keeps the math safe.
# * **Upper Bound (10):** We know biologically that low-dose radiation isn't a "death ray." Restricting it to 10 prevents the model from wandering off into unrealistic territory.



# ### 6. The Result: `res_linear`

# This variable stores the **"Final Report."**
# Once the code finishes running, `res_linear` will contain:

# * `res_linear.x`: The "Winning Numbers" (The best-fit parameters).
# * `res_linear.fun`: The lowest Error Score achieved.
# * `res_linear.success`: `True` if it worked, `False` if it failed.

# ============p values====================

# in statistical analysis (and in the context of these radiation papers), the **p-value** is not used *only* to compare the two models. It serves **two distinct and critical purposes**.

# In the script I provided, I explicitly calculated the p-value for **Purpose #1**. However, a full analysis (like Preston's) also relies heavily on **Purpose #2**.

# ### 1. The Model Comparison Test (Likelihood Ratio Test)

# * **Question:** *"Does the complex model (Curve) fit the data significantly better than the simple model (Line)?"*
# * **What it Tests:** It tests the **Gamma ()** parameter.
# * **The Logic:**
# * **Null Hypothesis:** Gamma is zero (The world is linear).
# * **P-Value < 0.05:** The Linear model is "broken" (misspecified). You *must* use the Curved model.
# * **P-Value > 0.05:** The Linear model is fine. Adding the curve didn't help enough to matter.


# * **In the Script:** This is the specific `p_value` variable calculated at the very end using `chi2.sf`.

# ### 2. The Significance of Risk Test (Wald Test)

# * **Question:** *"Is the radiation risk actually real, or is the slope just zero?"*
# * **What it Tests:** It tests the **Beta ()** parameter.
# * **The Logic:** Even if you decide the "Linear Model" is the correct shape, you still need to know if the line is **flat** or **slanted**.
# * **Null Hypothesis:** Beta is zero (Radiation is harmless).
# * **P-Value < 0.05:** The slope is real. Radiation causes mortality.
# * **P-Value > 0.05:** The slope is indistinguishable from zero. There is no detectable risk.


# * **In the Script:** My simple script did *not* automatically print this p-value (it requires calculating the Hessian/Standard Errors, which `minimize` doesn't output by default without extra configuration).

# ### Comparison: Which P-Value answers which question?

# | Scenario | **P-Value for Model Comparison (LRT)** | **P-Value for Slope (Beta)** | **Conclusion** |
# | --- | --- | --- | --- |
# | **A** | **High (> 0.05)** | **High (> 0.05)** | **"Robust Null"**<br>

# <br>The model is Linear, and the Line is flat. Radiation does nothing. (This matches the Wang/Oak Ridge draft results ).

#  |
# | **B** | **High (> 0.05)** | **Low (< 0.05)** | **"Linear Risk"**<br>

# <br>The model is Linear, but the line is slanted up. Radiation is harmful. |
# | **C** | **Low (< 0.05)** | **(Ignored)** | **"Non-Linear Risk"**<br>

# <br>The Linear model is wrong. You must switch to the Quadratic model and check its curve. |

# ### How to get the "Slope P-Value" in Python?

# If you want to know if the risk itself is significant (Purpose #2), you need the **Confidence Interval** or the **Standard Error** of Beta.

# In the Wang/Oak Ridge draft, they report this by looking at the **Confidence Interval (CI)** rather than just a p-value.

# * If the **95% CI includes Zero** (e.g., -0.2 to 0.5), it means **P > 0.05** (Not Significant).
# * The draft states: *"estimates... were, overall, not significantly different from zero"* and *"95% CIs encompassing zero"*.



# **Summary:**

# * **LRT P-Value (in script):** Decides the **Shape** (Straight vs. Curved).
# * **Wald P-Value (implied):** Decides the **Danger** (Safe vs. Harmful).

# You need **both** to write a complete report. If the Shape is Linear (LRT p > 0.05), you then look at the Beta P-value to see if that Linear Shape is actually rising.

 
### Why it wasn't there before

# The `minimize` function in Python is a general-purpose solver. Unlike specialized statistical software (like R or Stata), it finds the "best fit" numbers but doesn't automatically calculate the standard errors or p-values. We have to do the math ourselves.

### The Solution: Add this "Standard Error" Calculator
### Interpretation for Your Report
 

# 1. **Slope P-Value (from this new code):**
# * **Question:** "Is there a linear risk?"
# * **If P > 0.05:** You have confirmed the **Wang/Oak Ridge** finding. The traditional model agrees that there is no detectable linear risk.
# * **If P < 0.05:** The traditional model *claims* there is a risk. This sets up your conflict: "Why does the Parametric model see a risk when the Causal ML model (Wang) does not?"


# 2. **LRT P-Value (from the original script):**
# * **Question:** "Is there a curve?"
# * **If P > 0.05:** The simple linear model (even if null) is the best fit.
# * **If P < 0.05:** The linear model is wrong; you need the curve.



# **Summary:** The original script tested the **Shape**. This addition tests the **Significance**. You need both to do a "Systematic Comparison" as your professor asked.