import pandas as pd
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter, GeneralizedGammaFitter, ExponentialFitter
import matplotlib.pyplot as plt
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore")
# Load the dataset
telco_data = pd.read_csv('telco.csv')  # Replace with the correct path
def modify_data(input_data):
    processed_data = input_data.copy()
    processed_data = processed_data.set_index('ID')  # Assuming 'ID' is the index column

    # Convert categorical variables into numerical using label encoding
    categorical_cols = ['region', 'retire', 'marital', 'ed', 'gender', 'voice', 'internet', 'custcat', 'churn', 'forward']
    for col in categorical_cols:
        processed_data[col] = pd.factorize(processed_data[col])[0]

    return processed_data
telco_data_encoded = modify_data(telco_data)
telco_data_encoded
# Parametric Models
# Initialize the AFT models
aft_models = {
    "Weibull": WeibullAFTFitter(),
    "LogNormal": LogNormalAFTFitter(),
    "LogLogistic": LogLogisticAFTFitter(),
    "GeneralizedGamma": GeneralizedGammaFitter(),
    "Exponential": ExponentialFitter()
}

# Fit the models
for model_name, model in aft_models.items():
    try:
        # Adjusting fit method for GeneralizedGamma and Exponential models
        if model_name in ["GeneralizedGamma", "Exponential"]:
            model.fit(telco_data_encoded['tenure'], telco_data_encoded['churn'])
        else:
            model.fit(telco_data_encoded, 'tenure', 'churn')
        print(f"{model_name} AFT Model Fitted.")
    except Exception as e:
        print(f"Error fitting {model_name}: {e}")


# Print model summaries
for model_name, model in aft_models.items():
    try:
        print(f"{model_name} AFT Model Summary:")
        print(model.summary)
    except Exception as e:
        print(f"Error in summarizing {model_name}: {e}")
# Save the models
for model_name, model in aft_models.items():
    try:
        with open(f"{model_name}_aft_model.pkl", 'wb') as f:
            pickle.dump(model, f)
        print(f"{model_name} model saved.")
    except Exception as e:
        print(f"Error saving {model_name} model: {e}")
# After fitting models
for model_name, model in aft_models.items():
    try:
        # Use summary for model comparison
        print(f"{model_name} AFT Model Summary:")
        print(model.summary)
    except Exception as e:
        print(f"Error in summarizing {model_name}: {e}")
for model_name, model in aft_models.items():
    try:
        with open(f"{model_name}_aft_model.pkl", 'wb') as f:
            pickle.dump(model, f)
        print(f"{model_name} model saved.")
    except Exception as e:
        print(f"Error saving {model_name} model: {e}")

# Predicting median survival time for an Exponential distribution
if isinstance(model, ExponentialFitter):
    lambda_ = model.summary.loc['lambda_', 'coef']
    median_survival_time = np.log(2) / lambda_
    telco_data_encoded['PredictedMedianLifetime'] = median_survival_time
else:
    telco_data_encoded['PredictedMedianLifetime'] = model.predict_median(telco_data_encoded)


plt.figure(figsize=(10, 6))


# Plot survival functions for Weibull, LogNormal, and LogLogistic models only
desired_models = ["Weibull", "LogNormal", "LogLogistic"]

for model_name, model in aft_models.items():
    try:
        if model_name in desired_models and hasattr(model, 'predict_survival_function'):
            sample = telco_data_encoded.iloc[:10]  # Adjust as needed
            survival_function = model.predict_survival_function(sample)
            plt.plot(survival_function.mean(axis=1), label=f'{model_name} Survival Function')
    except Exception as e:
        print(f"Error in plotting for {model_name}: {e}")

plt.title('Survival Curves by Weibull, LogNormal, and LogLogistic AFT Models')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.legend()
plt.show()

# CLV
# List of columns considered significant
important_columns = ["address", "age", "internet", "marital", "tenure", "churn",
                       "custcat", "custcat", "voice"]

important_data = telco_data_encoded[important_columns]
# Fit the Log-Normal model
log_norm = aft_models["LogNormal"].fit(telco_data_encoded, duration_col='tenure', event_col='churn')
log_norm_prediction = log_norm.predict_survival_function(telco_data_encoded).T
log_norm_prediction_avg = log_norm_prediction.mean()
log_norm.print_summary()
# Creating a copy of Log-Normal model predictions for Customer Lifetime Value (CLV) calculation
clv_predictions = log_norm_prediction.copy()

# Setting up parameters for CLV calculation
margin_per_month = 1000  # Average monthly margin per customer
months_sequence = range(1, len(clv_predictions.columns) + 1)  # Sequence of months
discount_rate = 0.1  # Discount rate

# Adjusting the CLV data based on the discount rate
for month in months_sequence:
    clv_predictions.loc[:, month] = clv_predictions.loc[:, month] / ((1 + discount_rate / 12) ** (months_sequence[month - 1] - 1))
    
# Calculating CLV for each customer
clv_predictions["CLV"] = margin_per_month * clv_predictions.sum(axis=1)
clv_predictions
# Report
telco_data["CLV"] = clv_predictions.CLV

# Analyzing CLV based on different groupings.
print(telco_data.groupby(["gender", "ed", "marital"])[["CLV"]].mean())
print(telco_data.groupby("gender")[["CLV"]].mean())
print(telco_data.groupby("voice")[["CLV"]].mean())
print(telco_data.groupby("forward")[["CLV"]].mean())
print(telco_data.groupby("internet")[["CLV"]].mean())
print(telco_data.groupby("marital")[["CLV"]].mean())
print(telco_data.groupby("region")[["CLV"]].mean())
print(telco_data.groupby("custcat")[["CLV"]].mean())
print(telco_data.groupby("retire")[["CLV"]].mean())
print(telco_data.groupby("ed")[["CLV"]].mean())

## Budget Calculations
# Filter retained customers
telco_data_encoded["CLV"] = clv_predictions.CLV
retained_customers = telco_data_encoded["CLV"][telco_data_encoded['churn'] == 0]
# Calculate retained CLV
retained_clv = retained_customers.sum()

# Define retention rate and cost per customer
retention_rate = 0.8  # Assuming 80% retention rate
cost_per_customer = 5000  # Assuming $5000 cost per customer

# Calculate retention cost and annual budget
retention_cost = len(telco_data_encoded) * retention_rate * cost_per_customer
annual_budget = retained_clv - retention_cost

print("RETAINED CLV:", retained_clv)
print("RETENTION COST:", retention_cost)
print("ANNUAL BUDGET:", annual_budget)


