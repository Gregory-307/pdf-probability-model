import pandas as pd
from tqdm import tqdm
from observed_coefficients import calculate_observed_coefficients

def add_differences(data, columns=['value', 'quality', 'momentum', 'growth']):
    """
    Add difference columns for specified features in the dataset.

    Parameters:
        data (pd.DataFrame): DataFrame containing the raw data.
        columns (list): List of columns to calculate differences for.

    Returns:
        pd.DataFrame: DataFrame with added difference columns.
    """
    for col in columns:
        diff_col = f"{col}_diff"
        data[diff_col] = data.groupby('ticker')[col].diff()
    return data

def calculate_next_days_mean(data, column='close', days=10): ### Not efficient but pandas rolling requires monotonic date column
    """
    Manually calculate the mean of the next 10 days for a specified column, grouped by ticker.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        column (str): Column to calculate the mean for.
        days (int): Number of calendar days to look ahead.

    Returns:
        pd.DataFrame: DataFrame with an added column for the mean of the next 10 days.
    """
    # Ensure the data is sorted by ticker and date
    data = data.sort_values(by=['ticker', 'date']).reset_index(drop=True)

    def calculate_for_group(group):
        # Iterate through each row to calculate the mean manually
        group['next_days_mean'] = None
        for i in range(len(group)-1):
            current_date = group.iloc[i + 1]['date']

            # Filter data within the next 10 days for the current ticker group
            next_days = group[
                (group['date'] > current_date) &
                (group['date'] <= current_date + pd.Timedelta(days=days))
            ]

            # Calculate the mean for the column and assign it
            group.iloc[i, group.columns.get_loc('next_days_mean')] = next_days[column].mean() if not next_days.empty else None

        return group

    # Group by ticker and apply the calculation
    result = data.groupby('ticker', group_keys=False).apply(calculate_for_group)

    return result





def preprocess_data(
    raw_data_path="data/raw_data.csv",
    processed_data_path="data/processed_data.csv",
    column="close",
    horizon=60
):
    """
    Full preprocessing pipeline: calculate observed coefficients and save results.

    Parameters:
        raw_data_path (str): Path to the raw data CSV file.
        processed_data_path (str): Path to save the processed data CSV file.
        column (str): Column to process (e.g., "close").
        horizon (int): Number of days for observed coefficient calculation.

    Returns:
        None
    """
    # Step 1: Load and validate raw data
    print("Loading raw data...")
    data = pd.read_csv(raw_data_path, parse_dates=['date'], encoding="unicode_escape")
    data.columns = map(str.lower, data.columns)

    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d', dayfirst=False)

        
    if column.lower() not in data.columns:
        raise ValueError(f"Column '{column}' not found in the data.")

    # Step 2: Sort and clean data
    print("Sorting and cleaning data...")
    data.sort_values(by=['ticker', 'date'], inplace=True)
    data.dropna(inplace=True)

    # Step 3: Add differences
    print("Calculating feature differences...")
    data = add_differences(data)

    # Step 4: Calculate the mean of the next 10 days
    print("Calculating the mean of the next 10 days...")
    data = calculate_next_days_mean(data, column=column.lower())

    # Step 5: Group data by ticker and process each group
    # grouped = data.groupby('ticker')
    # processed_groups = []

    # print("Processing stocks...")
    # for ticker, group in tqdm(grouped, desc="Processing stocks"):
    #     # Sort by date
    #     group.sort_values(by='date', inplace=True)

    #     # Calculate observed coefficients
    #     group = calculate_observed_coefficients(
    #         group,
    #         column=column.lower(),
    #         horizon=horizon
    #     )
    #     processed_groups.append(group)

    # # Step 6: Combine all processed groups
    # print("Combining processed data...")
    # data = pd.concat(processed_groups, ignore_index=True)

    # Step 7: Save the processed data
    print(f"Saving processed data to {processed_data_path}...")
    data.to_csv(processed_data_path, index=False)
    print("Processing complete.")

if __name__ == "__main__":
    preprocess_data()