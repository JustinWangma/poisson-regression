
## script

Yes, the script you provided faithfully implements the **mathematical core** of the methods used in Preston et al. (2007) and Grant et al. (2017).

Here is a breakdown of exactly how the code maps to the specific equations and methods in those papers:

### 1. The Risk Model (Linear ERR)

The defining feature of the Preston/Grant approach is the **Excess Relative Risk (ERR)** model, which is additive (`1 + Risk`) rather than the exponential (`exp(Risk)`) model used in standard Poisson regression.

* **Preston Paper Equation:**
$$\lambda(d) = \lambda_0 \times (1 + \beta d)$$

* **Your Script:**
```python
rr = 1 + err_beta * data['dose_gy']
lambda_pred = bkg * rr * data['person_years']

```


* **Verdict:** **Exact Match.** This correctly implements the non-linear "1 + beta*dose" structure that prevents standard software from fitting these models easily.



### 2. The Shape Test (Linear vs. Linear-Quadratic)

Grant et al. (2017) focused on testing if the dose-response curve had significant curvature (a "J-shape" or upward bend).

* **Grant Paper Method:** Compare a Linear model () against a Linear-Quadratic model () using a Likelihood Ratio Test.
* **Your Script:**
```python
# Fits LQ Model
rr = 1 + err_beta * data['dose_gy'] + err_gamma * (data['dose_gy']**2)
# Calculates Test Statistic
d_nll = 2 * (res_linear.fun - res_lq.fun)
p_value_curvature = chi2.sf(d_nll, df=1)

```


* **Verdict:** **Exact Match.** This performs the standard Likelihood Ratio Test (LRT) with 1 degree of freedom, exactly as described in the methodology sections of those papers.



### 3. The Objective Function (Poisson Likelihood)

Both papers assume the number of deaths follows a Poisson distribution.

* **Statistical Theory:** Minimizing the Negative Log-Likelihood (NLL) is mathematically identical to maximizing the Likelihood, which is how parameters are estimated in these studies.
* **Your Script:**
```python
nll = np.sum(lambda_pred - data['deaths'] * np.log(lambda_pred))

```


* **Verdict:** **Exact Match.** This is the correct kernel for the Poisson loss function.



### A Note on Simplification

While the **math** is identical, the **data structure** in your script is simplified compared to the actual papers:

* **Papers:** They used **"Grouped Data"** (stratified person-year tables), where thousands of people are collapsed into rows like "Males, Age 40-45, Dose 0-10mGy."
* **Your Script:** Uses **"Individual Data"** (one row per person).
* **Does it matter?** **No.** For modern computing, individual data is actually *better* and more precise. The grouped method was used in 2007 largely to save computing power. The results will be statistically equivalent.

**Summary:** This script is a valid implementation of the "Preston/Grant Method" for use in your comparison against the Causal ML models.
## script's goal : dose_gy ==> count of deaths
Yes, exactly. To be precise, the script's goal is to **quantify** that relationship by finding the "best fitting" line (or curve) that connects `dose_gy` to the count of `deaths`.

Here is the simple translation of the script's objective:

### 1. The Core Question

The script asks: *"If I know a worker's radiation dose (`dose_gy`), how much does that multiply their chance of dying (`deaths`)?"*

### 2. The "Knobs" it Turns

To answer that, the computer tweaks the **`err_beta`** (and `err_gamma`) value up and down until the prediction matches reality.

* **If `err_beta` is High (e.g., 5.0):** The model predicts that even a small dose causes a lot of extra deaths.
* **If `err_beta` is Zero (0.0):** The model predicts that dose doesn't matter at all (the relationship is non-existent).

### 3. The Final Output

When the script finishes, it gives you that specific number (the slope).

* **Relationship Found:** "We found that for every 1 Gray of dose, deaths increase by 45% (`beta = 0.45`)."
* **No Relationship:** "We found the slope is effectively zero (`beta = 0.01`, p > 0.05). Dose does not predict deaths."

So, strictly speaking, you are using `deaths` (the outcome) and `dose_gy` (the input) to solve for `err_beta` (the relationship).

## how about variables like Person-Years , Sex and age

Great question. In the script and the Preston/Grant models, variables like **Person-Years**, **Sex**, and **Age** play specific, distinct roles. They are not just "more inputs"—they fit into different parts of the mathematical equation.

Here is exactly how each one functions in the model:

### 1. Person-Years: The "Exposure Time" (Multiplier)

* **Role:** This is the **Offset**. It converts the *Rate* (risk per year) into a *Count* (expected deaths).
* **The Logic:** If the annual risk is 1%, a person watched for 10 years has 10x the chance of dying compared to a person watched for 1 year.
* **In the Script:**
```python
# Rate * Time = Expected Count
lambda_pred = bkg * rr * data['time'] 

```


*(Note: In the simulated data script, `time` is the person-years).*

### 2. Age and Sex: The "Baseline Risk" (Covariates)

* **Role:** These define the **Normal Background Rate** (the "1" in the equation).
* **The Logic:** Cancer is naturally a disease of aging. A 70-year-old is much more likely to die than a 20-year-old, regardless of radiation. Men and women also have different baseline cancer rates.
* **In the Script:**
```python
# Defines the "Normal" risk without radiation
bkg = np.exp(intercept + b_age * Age + b_sex * Sex)

```


* **Why it's crucial (Confounding):** Older workers usually have **higher doses** (because they worked longer). If you don't mathematically subtract the "Age Effect," the model will look at the deaths in high-dose older people and blame the radiation, when it was actually just old age. This variable "cleans" the data so `err_beta` only measures radiation.

### 3. Dose: The "Excess Risk" (Exposure)

* **Role:** This is the **Effect of Interest**. It is the only variable that gets multiplied by your target parameter (`err_beta`).
* **In the Script:**
```python
# The multiplier ON TOP of the normal risk
rr = 1 + err_beta * Dose

```



### Summary Equation

The full model puts them together like this:

$$\text{Deaths} \approx \underbrace{\text{Time}}_{\text{Person-Years}} \times \underbrace{\exp(\text{Age} + \text{Sex})}_{\text{Baseline Risk}} \times \underbrace{(1 + \beta \times \text{Dose})}_{\text{Radiation Effect}}$$

* **You tune `Age/Sex**` to get the baseline right.
* **You use `Time**` to scale it to the study duration.
* **You tune `Beta**` to see if Radiation adds anything extra on top.