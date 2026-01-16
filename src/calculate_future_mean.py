import pandas as pd
import numpy as np

from future_mean import add_future_mean

def preprocess_data(raw_data_path="pivoted_data.csv", processed_data_path="data/processed_data.csv", group_column="COMPANYNAME
", days_column="date", price_column="close", future_mean_column="Future Mean", N=10):
    """
    Loads raw data, preprocesses it by sorting and converting datetimes, and calculates the future mean for a specified number of days.

    Args:
        raw_data_path (str): Path to the raw data CSV file.
        processed_data_path (str): Path to save the processed data CSV file.
        group_column (str): The name of the column to group data by.
        days_column (str): The name of the column containing datetime values.
        price_column (str): The name of the column containing price close values.
        future_mean_column (str): The name of the column to store the calculated future means.
        N (int): The number of days to consider for calculating the future mean.

    Returns:
        None
    """
    print("Loading raw data...")
    data = pd.read_csv(raw_data_path)

    print("Converting and sorting data...")
    data[days_column] = pd.to_datetime(data[days_column], format='%Y-%m-%d', dayfirst=False)
    data = data.sort_values(by=[group_column, days_column]).dropna()

    print("Calculating future means...")
    data = add_future_mean(data, group_column, days_column, price_column, future_mean_column, N)

    print(f"Saving processed data to {processed_data_path}...")
    data.to_csv(processed_data_path, index=False)
    print("Processing complete.")

if __name__ == "__main__":
    preprocess_data()

