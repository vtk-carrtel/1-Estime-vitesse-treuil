import pandas as pd
import glob
import os

# Define the path to the output directory
output_dir = "/Users/khacviettran/Documents/Localwork/todaylist/1-estimation-vitesse-treuil/output/"

# Define patterns for Leman and Bourget CSV files
leman_csv_pattern = os.path.join(output_dir, "result_Leman_*.csv")
bourget_csv_pattern = os.path.join(output_dir, "result_Bourget_*.csv")

# Get lists of Leman and Bourget CSV files
leman_csv_files = glob.glob(leman_csv_pattern)
bourget_csv_files = glob.glob(bourget_csv_pattern)

# Function to process a list of CSV files, combine them, and calculate statistics
def process_files(csv_files, lake_name):
    all_dataframes = []
    for f in csv_files:
        df = pd.read_csv(f)
        all_dataframes.append(df)

    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_csv_path = os.path.join(output_dir, f"combined_results_{lake_name}.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Combined CSV for {lake_name} saved to {combined_csv_path}")

        if not combined_df.empty:
            inc_slope_stats = calculate_statistics(combined_df, 'inc_slope')
            print(f"Statistics for {lake_name} inc_slope: Mean={inc_slope_stats[0]}, Std={inc_slope_stats[1]}, Min={inc_slope_stats[2]}, Max={inc_slope_stats[3]}")

            dec_slope_stats = calculate_statistics(combined_df, 'dec_slope')
            print(f"Statistics for {lake_name} dec_slope: Mean={dec_slope_stats[0]}, Std={dec_slope_stats[1]}, Min={dec_slope_stats[2]}, Max={dec_slope_stats[3]}")
        else:
            print(f"Combined DataFrame for {lake_name} is empty. No statistics calculated.")
    else:
        print(f"No CSV files found for {lake_name} to combine.")
    return combined_df if all_dataframes and not combined_df.empty else pd.DataFrame()

# Calculate statistics for inc_slope and dec_slope
def calculate_statistics(df, column):
    """Calculate statistics for a given column in the DataFrame."""
    mean = df[column].mean()
    std = df[column].std()
    min_val = df[column].min()
    max_val = df[column].max()
    return mean, std, min_val, max_val

# Process Leman files
print("\nProcessing Leman files:")
combined_leman_df = process_files(leman_csv_files, "Leman")

# Process Bourget files
print("\nProcessing Bourget files:")
combined_bourget_df = process_files(bourget_csv_files, "Bourget")

