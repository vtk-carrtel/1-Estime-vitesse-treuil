import pandas as pd
import glob
import os

# --- Configuration Constants ---
OUTPUT_DIR = "output"  # Use a relative path for the output directory
# Define patterns for Leman and Bourget CSV files based on OUTPUT_DIR
# Original script used "result_Leman_*.csv", refactored script 1 saves as "slopes_Leman_*.csv"
# For consistency with the refactored script 1, we should use "slopes_"
LEMAN_CSV_PATTERN = os.path.join(OUTPUT_DIR, "slopes_Leman_*.csv")
BOURGET_CSV_PATTERN = os.path.join(OUTPUT_DIR, "slopes_Bourget_*.csv")


# --- Helper Functions ---

def load_and_combine_csv_files(csv_files_pattern, lake_name):
    """
    Loads and combines multiple CSV files found by a glob pattern.
    Saves the combined DataFrame to a new CSV file.
    Returns the combined DataFrame or an empty DataFrame if no files are found or an error occurs.
    """
    csv_files = glob.glob(csv_files_pattern)
    if not csv_files:
        print(f"No CSV files found for {lake_name} matching pattern: {csv_files_pattern}")
        return pd.DataFrame()

    all_dataframes = []
    for f_path in csv_files:
        try:
            df = pd.read_csv(f_path)
            all_dataframes.append(df)
        except pd.errors.EmptyDataError:
            print(f"Warning: CSV file {f_path} is empty and will be skipped.")
        except Exception as e:
            print(f"Error reading CSV file {f_path}: {e}. Skipping this file.")

    if not all_dataframes:
        print(f"No data loaded for {lake_name} after attempting to read files.")
        return pd.DataFrame()

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Ensure OUTPUT_DIR exists before trying to save the combined file
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    combined_csv_path = os.path.join(OUTPUT_DIR, f"combined_slopes_{lake_name}.csv")
    try:
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Combined CSV for {lake_name} saved to: {combined_csv_path}")
    except Exception as e:
        print(f"Error saving combined CSV for {lake_name} to {combined_csv_path}: {e}")
        return pd.DataFrame() # Return empty if save fails, to prevent processing incorrect data

    return combined_df


def calculate_statistics(dataframe, column_name):
    """
    Calculate mean, std, min, and max for a given column in the DataFrame.
    Returns a tuple (mean, std, min_val, max_val).
    Returns (None, None, None, None) if the column doesn't exist or data is unsuitable.
    """
    if column_name not in dataframe.columns:
        print(f"Warning: Column '{column_name}' not found in DataFrame. Cannot calculate statistics.")
        return None, None, None, None
    
    if dataframe[column_name].isnull().all():
        print(f"Warning: Column '{column_name}' contains all NaN values. Cannot calculate statistics.")
        return None, None, None, None
        
    try:
        mean_val = dataframe[column_name].mean()
        std_val = dataframe[column_name].std()
        min_val = dataframe[column_name].min()
        max_val = dataframe[column_name].max()
        return mean_val, std_val, min_val, max_val
    except Exception as e:
        print(f"Error calculating statistics for column '{column_name}': {e}")
        return None, None, None, None


def print_slope_statistics(dataframe, lake_name, slope_type_prefix):
    """
    Calculates and prints statistics for a specific slope type (e.g., 'inc_slope', 'dec_slope').
    The column name is constructed using the slope_type_prefix + "_slope".
    It also expects an R² column named slope_type_prefix + "_r2".
    """
    slope_column = f"{slope_type_prefix}_slope"
    r2_column = f"{slope_type_prefix}_r2"

    if dataframe.empty:
        print(f"DataFrame for {lake_name} is empty. No statistics to print for {slope_type_prefix} slopes.")
        return

    print(f"\n--- Statistics for {lake_name} - {slope_type_prefix.upper()} Slopes ---")

    # Slope statistics
    mean_slope, std_slope, min_slope, max_slope = calculate_statistics(dataframe, slope_column)
    if mean_slope is not None: # Check if stats were successfully calculated
        print(f"  Slope ({slope_column}):")
        print(f"    Mean: {mean_slope:.3f} m/s")
        print(f"    Std Dev: {std_slope:.3f} m/s")
        print(f"    Min: {min_slope:.3f} m/s")
        print(f"    Max: {max_slope:.3f} m/s")
    else:
        print(f"  Slope ({slope_column}): Statistics could not be calculated.")

    # R² statistics (optional, but good to have if data exists)
    if r2_column in dataframe.columns:
        mean_r2, std_r2, min_r2, max_r2 = calculate_statistics(dataframe, r2_column)
        if mean_r2 is not None:
            print(f"  R² ({r2_column}):")
            print(f"    Mean: {mean_r2:.3f}")
            print(f"    Std Dev: {std_r2:.3f}")
            print(f"    Min: {min_r2:.3f}")
            print(f"    Max: {max_r2:.3f}")
    else:
        print(f"  R² ({r2_column}): Column not found, statistics not calculated.")
    
    print(f"  Number of data points: {len(dataframe)}")


# --- Main Processing Logic ---

def process_lake_data(csv_pattern, lake_name):
    """
    Main function to process data for a single lake.
    Loads data, then calculates and prints statistics for inc and dec slopes.
    """
    print(f"\nProcessing data for {lake_name}...")
    combined_df = load_and_combine_csv_files(csv_pattern, lake_name)

    if not combined_df.empty:
        # The column names from `estimate_vitesse_refactored.py` are like 'inc_slope', 'dec_slope'
        # and 'inc_r2', 'dec_r2'.
        print_slope_statistics(combined_df, lake_name, "inc") # For 'inc_slope' and 'inc_r2'
        print_slope_statistics(combined_df, lake_name, "dec") # For 'dec_slope' and 'dec_r2'
    else:
        print(f"No data processed for {lake_name}. Skipping statistics calculation.")


def main():
    """
    Main function to orchestrate the processing for all lakes.
    """
    # Ensure the main output directory exists (or is created by load_and_combine_csv_files if needed)
    # It's good practice to ensure it exists before starting specific processing.
    if not os.path.exists(OUTPUT_DIR):
        print(f"Output directory {OUTPUT_DIR} does not exist. It will be created if CSV files are found and combined.")

    process_lake_data(LEMAN_CSV_PATTERN, "Leman")
    process_lake_data(BOURGET_CSV_PATTERN, "Bourget")

    print("\n--- Processing finished ---")

if __name__ == "__main__":
    main()
