# Reproduction of Atomic Bomb Survivor Mortality Analysis (LSS Report 14)

##  script formulas
Yes, the formulas in your script are now **highly consistent** with the mathematical framework described in LSS Report 14, with one justifiable simplification regarding the background rate.

Here is the comparison between your code and the paper's methodology:

### 1. The Radiation Risk Model (ERR)

* **Your Code:** `rr = 1 + err_beta * dose_gy`
* 
**The Paper:** The paper defines the linear Excess Relative Risk (ERR) model as: .


* **Verdict:** **Exact Match.** You are correctly implementing the standard Linear ERR model.

### 2. The Attained Age Term (Power Model)

* **Your Code:** `b_age * np.log(data['age']/70.0)`
* 
**The Paper:** The paper uses the term  (log of attained age) when modeling age effects. Mathematically,  is equivalent to  (a Power Function).


* **Centering at 70:** Your choice to divide by 70 (`age/70.0`) aligns perfectly with the paper's reporting convention. The paper explicitly reports risks standardized to "attained age of 70 years".


* **Verdict:** **Mathematically Consistent.** You are correctly modeling the "Power of Age" relationship rather than a simple exponential one.

### 3. The Background Rate (Simplification)

* 
**The Paper:** The paper states that the background rate  "was modeled by **stratification** for the ERR model". This means they essentially calculated a separate baseline rate for every tiny subgroup (e.g., "Men, Age 60-65, Hiroshima") rather than fitting a single smooth curve.


* **Your Code:** You are using a **Parametric** background model (a smooth formula with intercepts and slopes).
```python
log_rate = intercept + b_age * np.log(...) + ...

```


* **Verdict:** **Acceptable Adaptation.** While the paper used stratification (which requires estimating thousands of parameters), your parametric approach is the standard way to approximate this in a simplified analysis. Using the **log of age** (Power Model) for the background is the best possible way to approximate their stratified baseline, as cancer incidence biologically follows a power law.

### 4. The Likelihood Function

* **Your Code:** You are minimizing `lambda_pred - data['solid'] * np.log(lambda_pred)`.
* 
**The Paper:** The paper specifies the use of "Poisson regression methods for grouped survival data".


* **Verdict:** **Exact Match.** Your formula represents the Poisson loss function (Negative Log Likelihood).

### Summary

Your script successfully translates the complex methodology of LSS Report 14 into a runnable Python script.

* ✅ **ERR Formula:** Matches.
* ✅ **Age Effect:** Matches the "Power Law" math mentioned in the paper.
* ✅ **Optimization:** Matches the Poisson regression method.

You are ready to run this analysis. The results should be very close to the standard values (ERR  0.40–0.50 per Gy).

 
 

Here is a formal presentation of your results, structured for an academic setting (e.g., a slide deck or a report section). It focuses on **validation**, comparing your specific Python output directly against the published findings in **LSS Report 14**.

---


## Results
### 1. Methodology

* **Data Source:** Life Span Study (LSS) Report 14 dataset (1950–2003).
* **Statistical Framework:** Poisson Regression for grouped survival data (Person-Year Tables).
* **Model Specification:**
* **Radiation Effect:** Linear Dose-Response Model ().
* **Background Rate:** Modeled using a Power Function for Attained Age () to align with LSS methodology.
* **Optimization:** Negative Log-Likelihood minimization via `scipy.optimize`.



---

### 2. Computational Results (Python Script)

The following parameters were estimated using the replication script:

| Parameter | Estimate | Std. Error | P-Value | Interpretation |
| --- | --- | --- | --- | --- |
| **Intercept** | **-5.34** | 0.030 | < 0.001 | Baseline log-rate at age 70 (Female, Nagasaki). |
| **Attained Age** | **4.39** | 0.056 | < 0.001 | Background cancer risk scales with the **4.4 power** of age. |
| **Age at Exposure** | **0.14** | 0.067 | 0.033 | Older age at exposure slightly increases baseline risk. |
| **Sex (Male)** | **0.64** | 0.019 | < 0.001 | Males have ~1.9x higher background mortality than females. |
| **City (Hiroshima)** | **-0.03** | 0.021 | 0.223 | No significant difference in background rate between cities. |
| **Radiation Risk** | **0.48** | 0.044 | < 0.001 | **Excess Relative Risk (ERR) per Gray.** |

---

### 3. Validation Against Published Literature

The replication results show a high degree of alignment with **LSS Report 14**.

#### A. Primary Endpoint: Radiation Risk (ERR/Gy)

* **Script Result:** `0.48` (SE 0.044)
* 
**Published Result:** `0.47` (95% CI: 0.38, 0.56) for all solid cancer.


* **Conclusion:** The Python model successfully reproduced the primary radiation risk estimate. The point estimate (0.48) lies nearly in the exact center of the published confidence interval.

#### B. Background Age Dependence

* **Script Result:** `4.39` (Coefficient for ).
* 
**Published Result:** The report indicates that the **Excess Absolute Risk (EAR)** scales with the **3.4 power** of age , while the **Excess Relative Risk (ERR)** declines with the **-0.86 power** of age.


* **Validation:** Since , the background power can be derived as:



* **Conclusion:** The script’s estimated background power of **4.39** is consistent with the derived biological trend in the paper (~4.26).

#### C. Sex Effects

* **Script Result:** Significant positive coefficient for males (`0.64`), implying higher background risk.
* 
**Published Result:** The report confirms that "background mortality rates of cancer were substantially higher in men than in women".


* **Conclusion:** The model correctly identifies male sex as a strong risk factor for background mortality.

---

### 4. Discussion & Interpretation

* **Radiation Dose-Response:** The analysis confirms a highly significant, positive association between radiation dose and solid cancer mortality ().
* **Consistency:** The switch to a **Power Model** for age () was critical for replicating the paper's findings. This confirms that cancer mortality in this cohort follows a power law relative to age, rather than a simple exponential curve.
* **Independence of City:** The analysis found no statistically significant difference in background mortality between Hiroshima and Nagasaki (), validating the report's approach of pooling both cities into a single dataset for robust estimation.

### 5. Final Conclusion

The Python script utilizing `scipy.optimize` and a Poisson likelihood framework has successfully reproduced the key findings of **LSS Report 14**. The estimated Excess Relative Risk (ERR) of **0.48/Gy** is effectively identical to the published value of **0.47/Gy**.

This establishes a validated baseline model for **Step 2** of the project: applying Causal Machine Learning estimators to the same dataset.

## plot

This plot is the visual proof that the linear model (the red line) actually matches the real-world data (the black dots). It confirms that assuming a "straight line" for radiation risk was a correct scientific choice.

Here is the breakdown of exactly what you are seeing and how the math works behind the scenes.

### 1. Reading the Plot

* **X-Axis (Radiation Dose):** This is the "Treatment."
* Left side (0 Gy): People with no radiation.
* Right side (2.5 Gy): People with very high radiation exposure.


* **Y-Axis (Relative Risk):** This is the "Danger Level."
* **1.0:** Normal risk (same as an unexposed person).
* **2.0:** Double the risk.


* **The Black Dots (Observed Risk):**
* These are the "Real World" data points.
* We calculated these **without** assuming a straight line. We essentially asked: *"Hey, for the group of people who got exactly 1.0 Gy, how much cancer did they actually get compared to the 0 Gy group?"*


* **The Error Bars (Vertical Lines):**
* These show our uncertainty.
* **Small bars (at low dose):** We have lots of data (thousands of survivors), so we are very sure about the risk.
* **Huge bars (at high dose):** Very few people survived high radiation (e.g., >2 Gy). Because the sample size is small, the statistical margin of error is massive.


* **The Red Line (Your Model):**
* This is the math formula you calculated in Step 1: .
* **The Insight:** Notice how the red line passes almost perfectly through the black dots? That proves your model is excellent.



---

### 2. How the "ERR per Bin" is Calculated (The Math)

You asked how we calculate the risk for those specific dots (the bins). We use a method called **Categorical Regression**.

Instead of fitting one single slope (), we treat every dose group as if it were a totally different city or sex. We calculate a separate "multiplier" for each group.

#### Step A: Create the Bins

We slice the survivors into groups based on their dose.

* **Bin 0 (Reference):** 0 - 0.005 Gy (The Control Group).
* **Bin 1:** 0.005 - 0.1 Gy
* ...
* **Bin 5:** 1.0 - 2.0 Gy

#### Step B: The "Multiplier" Formula

We run a Poisson regression, but the formula looks different. instead of `Dose * Beta`, we use `Is_In_Bin_X`.

* **For the Control Group (Bin 0):** All the "Bin" switches are OFF (0).
* 


* **For the 1.0 Gy Group (Bin 5):** The "Bin 5" switch is ON (1).
* 



#### Step C: Calculating Relative Risk (RR)

The Relative Risk is simply the rate of the exposed group divided by the rate of the control group.

**Example from your plot:**

* Look at the dot near **1.0 Gy**.
* The Y-value is roughly **1.5**.
* This means .
* **Interpretation:** People in the 1.0 Gy bin died of cancer **1.5 times more often** (50% more) than the people in the 0 Gy bin, after adjusting for age and sex.

### plot Summary 

*"The black dots represent a categorical analysis where we estimated the risk for each dose group independently, making no assumptions about the shape of the curve. The red line is our linear model (). The fact that the dots align with the line confirms that the Linear No-Threshold (LNT) model provides a robust fit to the data."*