# Survival-analyses-on-telco-data

This repository contains code for marketing analytics utilizing survival analysis and customer lifetime value (CLV) modeling.

## Overview

The project involves the analysis of customer churn using survival analysis techniques and the calculation of CLV to derive actionable insights for marketing strategies.

## Dataset

The analysis uses the 'telco.csv' dataset. The dataset comprises various customer attributes such as region, tenure, churn, and categorical features.

## Code Overview

The main script `MA_Homework3_Zhora_Stepanyan.ipynb` contains the following major components:

### Data Preprocessing

The data preprocessing involves:

- Encoding categorical variables into numerical using label encoding.
- Modifying the dataset to prepare it for survival analysis models.

### Survival Analysis

The code includes fitting different parametric survival models:

- Weibull AFT Model
- LogNormal AFT Model
- LogLogistic AFT Model
- Generalized Gamma Model
- Exponential Model

The models are fitted to predict the tenure or survival time until customer churn.

### Customer Lifetime Value (CLV) Calculation

The CLV is calculated using the LogNormal model. The code computes CLV for each customer based on their predicted survival functions.

### Insights and Recommendations

The analysis provides insights and actionable recommendations for marketing strategies based on the CLV scores and survival analysis results.

## Usage

1. Ensure you have the necessary libraries installed (`pandas`, `lifelines`, `matplotlib`).
2. Load the dataset 'telco.csv'.
3. Run the provided Python notebook `MA_Homework3_Zhora_Stepanyan.ipynb` to perform the analysis.

## Results and Findings

The analysis identifies significant factors impacting customer churn and CLV. Notably, the 'retire' variable shows a substantial difference in CLV, suggesting potential marketing strategies for targeting specific demographics.

## Conclusion

The conclusions drawn from the analysis stress the importance of tailored marketing strategies based on CLV scores and survival analysis results.

## Budget Calculations

The script includes budget calculations based on retention rates, costs, and annual budgets derived from the analysis.

Feel free to explore the Jupyter notebook for a detailed walkthrough of the analysis.
