import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configuration Constants ---
OUTPUT_DIR = "output"  # Relative path for the output directory
# Input file names consistent with the output of 2-stats-all_refactored.py
LEMAN_RESULTS_FILENAME = "combined_slopes_Leman.csv"
BOURGET_RESULTS_FILENAME = "combined_slopes_Bourget.csv"

# --- Helper Functions ---

def load_and_prepare_data(output_dir_path, leman_filename, bourget_filename):
    """
    Loads Leman and Bourget data from CSV files, adds a 'Lake' column to each,
    and concatenates them into a single DataFrame.

    Args:
        output_dir_path (str): The directory where the CSV files are located.
        leman_filename (str): The filename for Leman data.
        bourget_filename (str): The filename for Bourget data.

    Returns:
        pandas.DataFrame: A combined DataFrame with data from both lakes.
                          Returns an empty DataFrame if any file is not found or an error occurs.
    """
    leman_full_path = os.path.join(output_dir_path, leman_filename)
    bourget_full_path = os.path.join(output_dir_path, bourget_filename)

    try:
        leman_df = pd.read_csv(leman_full_path)
        bourget_df = pd.read_csv(bourget_full_path)
    except FileNotFoundError as e:
        print(f"Error: File not found. Details: {e}")
        print(f"Please ensure the following files exist in the '{output_dir_path}' directory:")
        print(f"  - {leman_filename}")
        print(f"  - {bourget_filename}")
        return pd.DataFrame() # Return empty DataFrame on error
    except pd.errors.EmptyDataError as e:
        print(f"Error: One of the CSV files is empty. Details: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred while reading CSV files: {e}")
        return pd.DataFrame()


    leman_df['Lake'] = 'Leman'
    bourget_df['Lake'] = 'Bourget'

    combined_df = pd.concat([leman_df, bourget_df], ignore_index=True)
    return combined_df


def _calculate_single_summary(dataframe, column_name, lake_name):
    """
    Calculates median, min, and max for a given column and lake.
    Helper for generate_summary_text_for_plot.
    """
    lake_specific_data = dataframe[dataframe['Lake'] == lake_name][column_name]
    if lake_specific_data.empty or lake_specific_data.isnull().all():
        return f"Summary for {column_name} in {lake_name}:\n  Data not available\n{'-'*30}\n"

    median_val = lake_specific_data.median()
    min_val = lake_specific_data.min()
    max_val = lake_specific_data.max()
    return (f"Summary for {column_name} in {lake_name}:\n"
            f"  Median: {median_val:.2f} m/s\n"
            f"  Min: {min_val:.2f} m/s\n"
            f"  Max: {max_val:.2f} m/s\n{'-'*30}\n")


def generate_summary_text_for_plot(combined_data_df):
    """
    Generates a formatted string containing summary statistics (median, min, max)
    for 'inc_slope' and 'dec_slope' for each lake present in the combined DataFrame.

    Args:
        combined_data_df (pandas.DataFrame): The combined DataFrame with 'Lake',
                                             'inc_slope', and 'dec_slope' columns.

    Returns:
        str: A formatted string with summary statistics.
    """
    if not isinstance(combined_data_df, pd.DataFrame) or combined_data_df.empty:
        return "No data available to generate summary."

    summary_text_parts = []
    lakes = combined_data_df['Lake'].unique()
    slope_columns = ['inc_slope', 'dec_slope'] # Expected column names for slopes

    for lake in sorted(lakes): # Sort to ensure consistent order
        for slope_col in slope_columns:
            if slope_col in combined_data_df.columns:
                summary_text_parts.append(_calculate_single_summary(combined_data_df, slope_col, lake))
            else:
                summary_text_parts.append(f"Column '{slope_col}' not found for {lake}.\n{'-'*30}\n")
    
    return "".join(summary_text_parts)


def create_slope_comparison_boxplots(combined_data_df, output_dir_path):
    """
    Creates and saves a figure with two boxplots comparing 'inc_slope' and 'dec_slope'
    by 'Lake', and includes a summary text block.

    Args:
        combined_data_df (pandas.DataFrame): The DataFrame containing the data to plot.
                                             Must include 'Lake', 'inc_slope', 'dec_slope'.
        output_dir_path (str): The directory where the plot image will be saved.
    """
    if not isinstance(combined_data_df, pd.DataFrame) or combined_data_df.empty:
        print("Combined data is empty. Cannot generate boxplots.")
        return

    # Ensure output directory exists
    if not os.path.exists(output_dir_path):
        try:
            os.makedirs(output_dir_path)
            print(f"Created output directory: {output_dir_path}")
        except OSError as e:
            print(f"Error creating output directory {output_dir_path}: {e}")
            return

    fig, axes = plt.subplots(1, 2, figsize=(15, 7)) # Slightly wider for text

    # Boxplot for inc_slope (Descent speed)
    if 'inc_slope' in combined_data_df.columns:
        combined_data_df.boxplot(column='inc_slope', by='Lake', ax=axes[0], grid=True)
        axes[0].set_title('Vitesse de descente (inc_slope)')
        axes[0].set_ylabel('Vitesse (m/s)')
        axes[0].set_xlabel('') # Remove default 'Lake' x-label from pandas boxplot
        axes[0].tick_params(axis='x', rotation=0) # Keep lake names horizontal
    else:
        axes[0].text(0.5, 0.5, "'inc_slope' data not available", ha='center', va='center')
        axes[0].set_title('Vitesse de descente (inc_slope)')


    # Boxplot for dec_slope (Ascent speed)
    if 'dec_slope' in combined_data_df.columns:
        combined_data_df.boxplot(column='dec_slope', by='Lake', ax=axes[1], grid=True)
        axes[1].set_title('Vitesse de remontée (dec_slope)')
        axes[1].set_ylabel('Vitesse (m/s)')
        axes[1].set_xlabel('') # Remove default 'Lake' x-label
        axes[1].tick_params(axis='x', rotation=0) # Keep lake names horizontal
    else:
        axes[1].text(0.5, 0.5, "'dec_slope' data not available", ha='center', va='center')
        axes[1].set_title('Vitesse de remontée (dec_slope)')


    plt.suptitle('Comparaison des Vitesses de Descente et Remontée (Léman vs. Bourget)', fontsize=14)

    # Generate and add summary text to the figure
    summary_text = generate_summary_text_for_plot(combined_data_df)
    # Place text on the right side of the figure.
    # fig.text coordinates are (0,0) bottom-left to (1,1) top-right of the figure.
    # x=0.99 places it near the right edge. y=0.5 centers it vertically.
    # ha="right" aligns the right side of the text box with x=0.99.
    # va="center" centers the text box vertically around y=0.5.
    # bbox creates a styled box around the text.
    fig.text(0.99, 0.5, summary_text, transform=fig.transFigure,
             ha="right", va="center", fontsize=8, wrap=False, # Smaller fontsize
             bbox=dict(boxstyle="round,pad=0.5", fc="ivory", alpha=0.7))

    # Adjust layout to prevent overlap and make space for the summary text.
    # rect=[left, bottom, right, top] in figure coordinates.
    # Reducing 'right' (e.g., to 0.80 or 0.75) makes space for the text box on the right.
    # Reducing 'top' (e.g., to 0.92 or 0.90) makes space for suptitle.
    plt.tight_layout(rect=[0.02, 0.02, 0.78, 0.92]) # Adjusted for text and suptitle

    plot_filename = 'slopes_comparison_boxplot.png'
    plot_full_path = os.path.join(output_dir_path, plot_filename)
    try:
        plt.savefig(plot_full_path, bbox_inches='tight') # bbox_inches='tight' can help with layout
        print(f"Boxplot with summary saved to: {plot_full_path}")
    except Exception as e:
        print(f"Error saving plot to {plot_full_path}: {e}")

    # Optionally display the plot
    # plt.show() # Uncomment if you want to display the plot interactively

# --- Main Execution ---

def main():
    """
    Main function to load data, generate summary, and create boxplots.
    """
    print("Starting boxplot generation process...")
    combined_data = load_and_prepare_data(OUTPUT_DIR, LEMAN_RESULTS_FILENAME, BOURGET_RESULTS_FILENAME)

    if combined_data.empty:
        print("Halting script as combined data is empty or could not be loaded.")
        return

    # Print summary to console (optional, as it's also on the plot)
    console_summary = generate_summary_text_for_plot(combined_data)
    print("\n--- Console Summary ---")
    print(console_summary)

    # Create and save the plot
    create_slope_comparison_boxplots(combined_data, OUTPUT_DIR)
    
    print("\nBoxplot generation process finished.")


if __name__ == "__main__":
    main()
