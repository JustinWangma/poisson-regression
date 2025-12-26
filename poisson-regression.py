"""Poisson regression utilities and a small mock-data demo."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2


# ==========================================
# 1. MOCK DATA GENERATION (Simulating Oak Ridge)
# ==========================================
def generate_mock_data(n=11299, seed: int = 42):
    np.random.seed(seed)

    # Covariates
    age_entry = np.random.normal(25, 5, n)
    follow_up_years = np.random.uniform(5, 40, n)
    attained_age = age_entry + follow_up_years
    sex = np.random.binomial(1, 0.75, n)  # 75% Male (Wang draft)

    # Dose (Right-skewed, mean ~18mSv)
    dose_msv = np.random.lognormal(mean=2.5, sigma=1.2, size=n)
    dose_gy = dose_msv / 1000.0

    # Baseline Risk (increases with age, higher for men)
    baseline_risk = np.exp(-5 + 0.05 * attained_age + 0.3 * sex)
    radiation_multiplier = 1 + 0.5 * dose_gy

    lambda_rate = baseline_risk * radiation_multiplier
    expected_deaths = lambda_rate * follow_up_years

    deaths = np.random.poisson(expected_deaths)

    df = pd.DataFrame(
        {
            "person_years": follow_up_years,
            "attained_age": attained_age,
            "sex": sex,
            "dose_gy": dose_gy,
            "deaths": deaths,
        }
    )
    return df


# ==========================================
# 2. DEFINE THE MODELS (Grant et al. Style)
# ==========================================
def background_rate(params, data):
    intercept, b_age, b_sex = params[0], params[1], params[2]
    return np.exp(intercept + b_age * data["attained_age"] + b_sex * data["sex"])


def neg_log_likelihood_linear(params, data):
    bkg = background_rate(params, data)
    err_beta = params[3]
    rr = 1 + err_beta * data["dose_gy"]
    rr = np.maximum(rr, 1e-9)
    lambda_pred = bkg * rr * data["person_years"]
    nll = np.sum(lambda_pred - data["deaths"] * np.log(lambda_pred))
    return nll


def neg_log_likelihood_lq(params, data):
    bkg = background_rate(params, data)
    err_beta = params[3]
    err_gamma = params[4]
    rr = 1 + err_beta * data["dose_gy"] + err_gamma * (data["dose_gy"] ** 2)
    rr = np.maximum(rr, 1e-9)
    lambda_pred = bkg * rr * data["person_years"]
    nll = np.sum(lambda_pred - data["deaths"] * np.log(lambda_pred))
    return nll


# ==========================================
# 3. FIT AND COMPARE
# ==========================================
def fit_models(df):
    # Initial guesses
    init_linear = [-5, 0.05, 0.3, 0.1]

    res_linear = minimize(
        neg_log_likelihood_linear,
        init_linear,
        args=(df,),
        method="L-BFGS-B",
        bounds=[(None, None), (None, None), (None, None), (-2, 10)],
    )

    init_lq = list(res_linear.x) + [0.01]

    res_lq = minimize(
        neg_log_likelihood_lq,
        init_lq,
        args=(df,),
        method="L-BFGS-B",
        bounds=[(None, None), (None, None), (None, None), (-2, 10), (-5, 5)],
    )

    # Likelihood Ratio Test
    d_nll = 2 * (res_linear.fun - res_lq.fun)
    p_value = chi2.sf(d_nll, df=1)

    results = {
        "linear": res_linear,
        "lq": res_lq,
        "lr_stat": d_nll,
        "lr_pvalue": p_value,
    }
    return results


if __name__ == "__main__":
    df_mock = generate_mock_data()
    # Print first mock rows for quick inspection
    print(df_mock.head().to_string(index=False))
    results = fit_models(df_mock)
    print("Linear beta:", results["linear"].x[3])
    print("LQ beta:", results["lq"].x[3], "gamma:", results["lq"].x[4])
