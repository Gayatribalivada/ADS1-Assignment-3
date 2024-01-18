# Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from lmfit import Model

def read_clean_transpose_csv(csv_file_path):
    """
    Reads data from a CSV file, cleans the data, and returns the original,
    cleaned, and transposed data.

    Parameters:
    - csv_file_path (str): Path to the CSV file.

    Returns:
    - original_data (pd.DataFrame): Original data read from the CSV.
    - cleaned_data (pd.DataFrame): Data after cleaning and imputation.
    - transposed_data (pd.DataFrame): Transposed data.
    """

    # Read the data from the CSV file
    original_data = pd.read_csv(csv_file_path)

    # Replace non-numeric values with NaN
    original_data.replace('..', np.nan, inplace=True)

    # Select relevant columns
    columns_of_interest = [
        "CO2 emissions (metric tons per capita)",
        "CO2 emissions from solid fuel consumption (% of total)",
        "CO2 emissions from liquid fuel consumption (% of total)",
        "CO2 emissions from gaseous fuel consumption (% of total)",
        "GDP per capita growth (annual %)"
    ]

    # Create a SimpleImputer instance with strategy='mean'
    imputer = SimpleImputer(strategy='mean')

    # Apply imputer to fill missing values
    cleaned_data = original_data.copy()
    cleaned_data[columns_of_interest] = imputer.fit_transform(cleaned_data[columns_of_interest])

    # Transpose the data
    transposed_data = cleaned_data.transpose()

    return original_data, cleaned_data, transposed_data


def exponential_growth_model(x, a, b):
    """
    Exponential growth model function.

    Parameters:
    - x (array-like): Input values (time points).
    - a (float): Amplitude parameter.
    - b (float): Growth rate parameter.

    Returns:
    - array-like: Exponential growth model values.
    """
    return a * np.exp(b * np.array(x))


# The curvefitPlot function
def curvefitPlot(time_data, co2_emissions_data, result):
    """
    Plot the actual data, fitted curve, and confidence interval.

    Parameters:
    - time_data (array-like): Time points.
    - co2_emissions_data (array-like): Actual data values.
    - result (lmfit.model.ModelResult): Result of the curve fitting.

    Returns:
    None
    """

    plt.figure(figsize=(12, 8))  # Adjust figure size

    # Scatter plot for actual data
    sns.scatterplot(x=time_data, y=co2_emissions_data, label='Actual Data', color='blue', s=80)

    # Line plot for the exponential growth fit
    sns.lineplot(x=time_data, y=result.best_fit, label='Exponential Growth Fit', color='orange', linewidth=2)

    # Confidence interval plot
    plt.fill_between(time_data, result.best_fit - result.eval_uncertainty(), result.best_fit + result.eval_uncertainty(),
                     color='orange', alpha=0.2, label='Confidence Interval')

    plt.xlabel('Time')
    plt.ylabel('CO2 emissions (metric tons per capita)')
    plt.ylim(0, 30)  # Set y-axis range to [0, 30]
    plt.title('Curve Fit for CO2 emissions Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()
    
# Main Code

csv_file_path = "b2ef28b3-b5da-484e-b06f-7a644c838c7d_Data.csv"
original_data, cleaned_data, transposed_data = read_clean_transpose_csv(csv_file_path)

scaler = StandardScaler()
columns_of_interest = [
    "CO2 emissions (metric tons per capita)",
    "CO2 emissions from solid fuel consumption (% of total)",
    "CO2 emissions from liquid fuel consumption (% of total)",
    "CO2 emissions from gaseous fuel consumption (% of total)",
    "GDP per capita growth (annual %)"
]
df_normalized = scaler.fit_transform(cleaned_data[columns_of_interest])

kmeans = KMeans(n_clusters=3, random_state=42)
cleaned_data['Cluster'] = kmeans.fit_predict(df_normalized)

cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

silhouette_avg = silhouette_score(df_normalized, cleaned_data['Cluster'])
print(f"Silhouette Score: {silhouette_avg:.4f}")

plt.figure(figsize=(12, 8))
sns.scatterplot(x="GDP per capita growth (annual %)",
                y="CO2 emissions (metric tons per capita)",
                hue="Cluster", palette="viridis", data=cleaned_data, s=80)
sns.scatterplot(x=cluster_centers[:, 1], y=cluster_centers[:, 4],
                marker='X', s=200, color='red', label='Cluster Centers')
plt.title('Clustering of Countries with Cluster Centers')
plt.xlabel('GDP per capita growth (annual %)')
plt.ylabel('CO2 emissions')
plt.legend()
plt.show()

time_data = cleaned_data['Time']
co2_emissions_data = cleaned_data['CO2 emissions (metric tons per capita)']

model = Model(exponential_growth_model)
params = model.make_params(a=1, b=0.001)
result = model.fit(co2_emissions_data, x=time_data, params=params)

# Plot the curve fit for the entire dataset
curvefitPlot(time_data, co2_emissions_data, result)

# Generate time points for prediction for the entire dataset
future_years = [2024, 2027, 2030]

# Predict values for the future years using the fitted model for the entire dataset
predicted_values = result.eval(x=np.array(future_years))

# Display the predicted values for the entire dataset
for year, value in zip(future_years, predicted_values):
    print(f"Predicted value for {year} is : {value:.2f}")

# Filter data for China
china_data = cleaned_data[cleaned_data['Country Name'] == 'China']

# Extract relevant data for China
time_data_china = china_data['Time']
co2_emissions_china = china_data['CO2 emissions (metric tons per capita)']

# Predict values for China's CO2 emissions from 1990 to 2030 using the fitted model for the entire dataset
future_years_china = list(range(1990, 2031))
predicted_values_china = result.eval(x=np.array(future_years_china))

# Plot the CO2 emissions for the entire dataset
plt.figure(figsize=(12, 8))
sns.lineplot(x=time_data, y=co2_emissions_data, label='CO2 Emissions (Actual)', color='blue', linewidth=2)
plt.plot(future_years, predicted_values, label='Predicted Values', color='orange', linestyle='--', linewidth=2)
plt.xlabel('Time')
plt.ylabel('CO2 emissions (metric tons per capita)')
plt.title('CO2 Emissions Over Time with Predictions')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Plot the CO2 emissions for China and the predicted values from 1990 to 2030
plt.figure(figsize=(12, 8))
sns.lineplot(x=time_data_china, y=co2_emissions_china, label='CO2 Emissions for China (Actual)', color='green', linewidth=2)
plt.plot(future_years_china, predicted_values_china, label='Predicted Values', color='red', linestyle='--', linewidth=2)
plt.xlabel('Time')
plt.ylabel('CO2 emissions (metric tons per capita)')
plt.title('CO2 Emissions for China Over Time with Predictions (1990-2030)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Filter data for Canada
canada_data = cleaned_data[cleaned_data['Country Name'] == 'Canada']

# Extract relevant data for Canada
time_data_canada = canada_data['Time']
co2_emissions_canada = canada_data['CO2 emissions (metric tons per capita)']

# Predict values for Canada's CO2 emissions from 1990 to 2030 using the fitted model for the entire dataset
predicted_values_canada = result.eval(x=np.array(future_years_china))

# Plot the CO2 emissions for Canada and the predicted values from 1990 to 2030
plt.figure(figsize=(12, 8))
sns.lineplot(x=time_data_canada, y=co2_emissions_canada, label="CO2 Emissions for Canada (Actual)", color='purple', linewidth=2)
plt.plot(future_years_china, predicted_values_canada, label='Predicted Values', color='orange', linestyle='--', linewidth=2)
plt.xlabel('Time')
plt.ylabel('CO2 emissions (metric tons per capita)')
plt.title('CO2 Emissions for Canada Over Time with Predictions (1990-2030)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
