import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# --- Configuration Constants ---
OUTPUT_DIR = "output"
FOLDER_LEMAN = "data/Sonde_Profondeur-Leman/"
FOLDER_BOURGET = "data/Sonde_Profondeur-Bourget/"
DEFAULT_HEADER_PATTERN = "Date and Time;Seconds     ;Depth (m)                               ;"
R2_WINDOW_SIZE_FRACTION = 4  # Window size for R² calculation as a fraction of the segment length
SLOPE_DETECTION_EPSILON = 0.05
SLOPE_DETECTION_WINDOW = 201


# --- Helper Functions ---
def _ensure_output_dir_exists(output_dir):
    """Ensure the output directory exists."""
    os.makedirs(output_dir, exist_ok=True)


def find_header_line(file_path, pattern=DEFAULT_HEADER_PATTERN):
    """Finds the header line in a file based on a pattern."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if pattern in line:
                return i, line.rstrip('\n')
    raise ValueError(f"Header pattern {pattern!r} not found in {file_path}")


def get_lake_name(file_path):
    """Extracts lake name from file path."""
    if "Sonde_Profondeur-Leman" in file_path:
        return "Leman"
    elif "Sonde_Profondeur-Bourget" in file_path:
        return "Bourget"
    return "Unknown"


def _calculate_runs(mask):
    """Helper function to find consecutive runs of True in a boolean mask."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.empty((0, 2), dtype=int)
    splits = np.where(np.diff(idx) != 1)[0] + 1
    starts = np.insert(idx[splits], 0, idx[0])
    ends = np.append(idx[splits - 1], idx[-1])
    return np.vstack((starts, ends)).T


def find_most_linear_segment(time_segment, depth_segment, window_size_fraction=R2_WINDOW_SIZE_FRACTION):
    """
    Finds the most linear segment within a given time/depth segment using a sliding window
    and R² metric. The window_size is a fraction of the total segment length.
    """
    if len(time_segment) < 2: # Cannot fit a line to less than 2 points
        return 0, len(time_segment), -1

    window_size = max(2, round(len(time_segment) / window_size_fraction)) # Ensure window_size is at least 2

    if window_size >= len(time_segment): # If window is as large as segment, use whole segment
        window_size = len(time_segment) -1 # Adjust to ensure at least 2 points for polyfit

    if window_size < 2 : # if segment is too small
         coef = np.polyfit(time_segment, depth_segment, 1)
         pred = np.polyval(coef, time_segment)
         if np.sum((depth_segment - np.mean(depth_segment))**2) == 0: # Avoid division by zero if depth is constant
             r2 = 1 if np.allclose(depth_segment - pred, 0) else 0
         else:
             r2 = 1 - (np.sum((depth_segment - pred)**2) /
                       np.sum((depth_segment - np.mean(depth_segment))**2))
         return 0, len(time_segment), r2


    best_r2 = -1
    best_start_index = 0
    best_end_index = 0

    for i in range(len(time_segment) - window_size + 1):
        current_time_window = time_segment[i:i + window_size]
        current_depth_window = depth_segment[i:i + window_size]

        if len(current_time_window) < 2: # Should not happen with guard above but as safety
            continue

        # Fit linear regression
        coef = np.polyfit(current_time_window, current_depth_window, 1)
        pred = np.polyval(coef, current_time_window)

        # Calculate R²
        sum_sq_residuals = np.sum((current_depth_window - pred)**2)
        sum_sq_total = np.sum((current_depth_window - np.mean(current_depth_window))**2)

        if sum_sq_total == 0: # Avoid division by zero if depth is constant in window
            r2 = 1 if sum_sq_residuals == 0 else 0 # Perfect fit if residuals are also zero
        else:
            r2 = 1 - (sum_sq_residuals / sum_sq_total)

        if r2 > best_r2:
            best_r2 = r2
            best_start_index = i
            best_end_index = i + window_size
            
    return best_start_index, best_end_index, best_r2


def _analyze_segment(time_data, depth_data, segment_indices, smoothing_window_size):
    """
    Analyzes a specific segment (e.g., ascent or descent) to find the most linear part
    and calculate its slope.
    """
    i0, i1 = segment_indices
    # Extend window slightly for R2 analysis to ensure we capture the full slope
    # The smoothing_window_size is used as a proxy for how much to extend
    start_idx = max(0, i0 - smoothing_window_size)
    end_idx = min(len(time_data) -1, i1 + 1 + smoothing_window_size)

    segment_time = time_data[start_idx:end_idx]
    segment_depth = depth_data[start_idx:end_idx]

    if len(segment_time) < 2: # Not enough points for analysis
        return None, -1, -1 # Slope, R2, original indices for start/end

    # Find the most linear part of this segment
    best_sub_start_idx, best_sub_end_idx, r2_value = find_most_linear_segment(segment_time, segment_depth)

    linear_time = segment_time[best_sub_start_idx:best_sub_end_idx]
    linear_depth = segment_depth[best_sub_start_idx:best_sub_end_idx]

    if len(linear_time) < 2: # Not enough points for polyfit
        return None, r2_value, (time_data[i0], time_data[i1])


    coef = np.polyfit(linear_time, linear_depth, 1)
    slope = coef[0]
    
    # Return slope, R2, and original start/end times of the *detected* run (not the linear subsegment)
    return slope, r2_value, (time_data[i0], time_data[i1], depth_data[i0], depth_data[i1])


# --- Core Data Processing Functions ---

def load_and_prepare_data(file_path, header_pattern=DEFAULT_HEADER_PATTERN):
    """
    Handles finding the header, reading CSV, and basic data cleaning.
    Returns a pandas DataFrame with 'Time' and 'Depth'.
    """
    header_line_num, _ = find_header_line(file_path, header_pattern)
    try:
        data = pd.read_csv(file_path, sep=';', skiprows=header_line_num,
                           encoding='utf-8', on_bad_lines='skip') # Changed error_bad_lines to on_bad_lines
    except Exception as e:
        raise ValueError(f"Could not read CSV data from {file_path} after header: {e}")


    # Assuming the relevant columns are the second and third as per original logic
    # (Seconds and Depth)
    if data.shape[1] < 3:
        raise ValueError(f"File {file_path} does not have enough columns. Expected at least 3, got {data.shape[1]}.")

    # Drop the third data column (index 2) if it's the one with text, then select time and depth
    # The original script used 'Date', 'Time', 'Depth' after dropping a column.
    # Let's try to be more robust. Find 'Seconds' and 'Depth (m)'
    
    # Attempt to find columns by specific names after header processing
    # The header found by find_header_line is not used by read_csv to name columns if skiprows is used
    # So, we read column names from the header line itself.
    header_names = pd.read_csv(file_path, sep=';', skiprows=header_line_num, nrows=0, encoding='utf-8').columns
    data = pd.read_csv(file_path, sep=';', skiprows=header_line_num + 1, names=header_names,
                       encoding='utf-8', on_bad_lines='skip')


    time_col_name = next((col for col in data.columns if 'Seconds' in col), None)
    depth_col_name = next((col for col in data.columns if 'Depth (m)' in col), None)
    date_col_name = next((col for col in data.columns if 'Date and Time' in col), None)


    if not time_col_name or not depth_col_name or not date_col_name:
        raise ValueError(f"Could not find required columns ('Seconds', 'Depth (m)', 'Date and Time') in {file_path}")

    # Keep only necessary columns and rename them
    # Convert time and depth to numeric, coercing errors
    # It's important to handle non-numeric values that might appear
    data_cleaned = data[[date_col_name, time_col_name, depth_col_name]].copy()
    data_cleaned.columns = ['DateTime', 'Time', 'Depth']

    data_cleaned['Time'] = pd.to_numeric(data_cleaned['Time'], errors='coerce')
    data_cleaned['Depth'] = pd.to_numeric(data_cleaned['Depth'], errors='coerce')

    # Drop rows where 'Time' or 'Depth' could not be converted (became NaN)
    data_cleaned.dropna(subset=['Time', 'Depth'], inplace=True)
    
    if data_cleaned.empty:
        raise ValueError(f"No valid numeric data for Time and Depth found in {file_path}")

    return data_cleaned


def calculate_slopes_and_r2(time, depth, epsilon=SLOPE_DETECTION_EPSILON, smoothing_window_val=SLOPE_DETECTION_WINDOW):
    """
    Detects ascent (inc) and descent (dec) slopes in depth data.
    Returns a dictionary with slope analysis results and the smoothed depth array.
    """
    time_np = np.asarray(time, dtype=float)
    depth_np = np.asarray(depth, dtype=float)

    # Smooth depth data
    kernel = np.ones(smoothing_window_val) / smoothing_window_val
    depth_smoothed = np.convolve(depth_np, kernel, mode='same')

    # Calculate derivative
    derivative = np.gradient(depth_smoothed, time_np)
    positive_slope_mask = derivative > epsilon
    negative_slope_mask = derivative < -epsilon

    # Find runs of positive and negative slopes
    positive_runs = _calculate_runs(positive_slope_mask)
    negative_runs = _calculate_runs(negative_slope_mask)

    results = {}

    if positive_runs.size > 0:
        # Assume the longest positive run is the main descent (depth increasing)
        run_lengths = np.array([run[1] - run[0] for run in positive_runs])
        main_descent_run_indices = positive_runs[np.argmax(run_lengths)]
        
        slope, r2, (t_start, t_end, d_start, d_end) = _analyze_segment(time_np, depth_smoothed, main_descent_run_indices, smoothing_window_val)
        if slope is not None:
            results['inc'] = { # 'inc' traditionally means increasing depth (descent)
                'start_time': float(t_start), 'end_time': float(t_end),
                'start_depth': float(d_start), 'end_depth': float(d_end),
                'slope': slope, 'r2': r2
            }

    if negative_runs.size > 0 and 'inc' in results:
        # Find the first significant ascent *after* the main descent ends
        # Consider runs that start after the identified descent phase
        possible_ascent_runs = negative_runs[negative_runs[:, 0] > main_descent_run_indices[1]]
        if possible_ascent_runs.size > 0:
            # Take the first such run as the main ascent
            # Or, one could also take the longest among these
            main_ascent_run_indices = possible_ascent_runs[0] # Simplistic: take the first one
            # Alternative: take the longest ascent after descent
            # run_lengths = np.array([run[1] - run[0] for run in possible_ascent_runs])
            # main_ascent_run_indices = possible_ascent_runs[np.argmax(run_lengths)]

            slope, r2, (t_start, t_end, d_start, d_end) = _analyze_segment(time_np, depth_smoothed, main_ascent_run_indices, smoothing_window_val)
            if slope is not None:
                results['dec'] = { # 'dec' traditionally means decreasing depth (ascent)
                    'start_time': float(t_start), 'end_time': float(t_end),
                    'start_depth': float(d_start), 'end_depth': float(d_end),
                    'slope': slope, 'r2': r2
                }
    
    return results, depth_smoothed


def save_slope_results(slope_results_dict, output_dir, lake_name, date_str):
    """Saves the slope results to a CSV file."""
    _ensure_output_dir_exists(output_dir)
    output_csv_path = os.path.join(output_dir, f"slopes_{lake_name}_{date_str}.csv")
    
    # Flatten the dictionary for CSV output
    flat_results = {}
    for key, value_dict in slope_results_dict.items():
        if isinstance(value_dict, dict):
            for sub_key, sub_value in value_dict.items():
                flat_results[f"{key}_{sub_key}"] = sub_value
        else:
            flat_results[key] = value_dict
            
    df = pd.DataFrame([flat_results])
    df.to_csv(output_csv_path, index=False)
    print(f"Saved slope results to {output_csv_path}")


def generate_and_save_plot(time, depth, smoothed_depth, slope_results,
                           output_dir, lake_name, date_str, file_path_for_title):
    """Handles generating and saving the plot."""
    _ensure_output_dir_exists(output_dir)
    output_plot_path = os.path.join(output_dir, f"plot_{lake_name}_{date_str}.png")

    plt.figure(figsize=(12, 7))
    plt.plot(time, depth, label="Raw Depth", alpha=0.7)
    plt.plot(time, smoothed_depth, linewidth=1.5, label="Smoothed Depth")

    # Plotting and annotating descent (inc) and ascent (dec) slopes
    for phase, color, label_prefix in [('inc', 'red', 'Descent'), ('dec', 'green', 'Ascent')]:
        if phase in slope_results and slope_results[phase]:
            data = slope_results[phase]
            plt.axvspan(data['start_time'], data['end_time'], color=color, alpha=0.15, label=f"{label_prefix} Phase")
            
            # Plot the identified linear segment used for slope calculation
            # This requires the actual time/depth points of that linear segment,
            # which are not directly in slope_results. For now, draw line between phase start/end.
            plt.plot([data['start_time'], data['end_time']],
                     [data['start_depth'], data['end_depth']],
                     color=color, linestyle='--', linewidth=2,
                     label=f"{label_prefix} Slope ({data['slope']:.2f} m/s, R²: {data.get('r2', -1):.2f})")

            # Annotation
            ymin, ymax = plt.ylim()
            y_annotate = (ymin + ymax) / 2
            plt.annotate(
                f"{label_prefix} Rate: {data['slope']:.2f} m/s\nR²: {data.get('r2', -1):.2f}",
                xy=(data['start_time'], data['start_depth']),
                xytext=(data['start_time'] + (data['end_time'] - data['start_time'])*0.1, y_annotate),
                ha='left', va='center', color=color,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color=color)
            )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Depth (meters)")
    plt.title(f"Depth Profile Analysis: {os.path.basename(file_path_for_title)}\nDate: {date_str} - Lake: {lake_name}")
    plt.legend(loc='best')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_plot_path, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    print(f"Saved plot to {output_plot_path}")


def process_file_refactored(file_path, output_dir=OUTPUT_DIR):
    """
    Refactored processing pipeline for a single data file.
    """
    try:
        print(f"Processing {file_path}...")
        lake_name = get_lake_name(file_path)
        
        # 1. Load and Prepare Data
        prepared_data_df = load_and_prepare_data(file_path)
        time_data = prepared_data_df['Time'].values
        depth_data = prepared_data_df['Depth'].values
        # Assuming sampling date is the first date entry in the 'DateTime' column
        sampling_date_str = pd.to_datetime(prepared_data_df['DateTime'].iloc[0]).strftime("%Y-%m-%d_%H%M%S")


        # 2. Calculate Slopes
        # Epsilon and window are now constants or passed as arguments
        slope_analysis_results, smoothed_depth = calculate_slopes_and_r2(
            time_data, depth_data,
            epsilon=SLOPE_DETECTION_EPSILON,
            smoothing_window_val=SLOPE_DETECTION_WINDOW
        )

        # Add file info to results if not already there
        slope_analysis_results['file'] = os.path.basename(file_path)
        slope_analysis_results['sampling_date'] = sampling_date_str
        slope_analysis_results['lake'] = lake_name

        # 3. Save Slope Results
        if slope_analysis_results.get('inc') or slope_analysis_results.get('dec'):
            save_slope_results(slope_analysis_results, output_dir, lake_name, sampling_date_str)
        else:
            print(f"No significant slopes found for {file_path}. CSV not saved.")

        # 4. Generate and Save Plot
        generate_and_save_plot(time_data, depth_data, smoothed_depth,
                               slope_analysis_results, output_dir,
                               lake_name, sampling_date_str, file_path)
        print(f"Successfully processed {file_path}")

    except ValueError as ve:
        print(f"Value error processing {file_path}: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred processing {file_path}: {e}")


def main():
    _ensure_output_dir_exists(OUTPUT_DIR)

    files_leman = [os.path.join(FOLDER_LEMAN, f) for f in os.listdir(FOLDER_LEMAN)
                   if f.endswith('.csv') and os.path.isfile(os.path.join(FOLDER_LEMAN, f))]
    files_bourget = [os.path.join(FOLDER_BOURGET, f) for f in os.listdir(FOLDER_BOURGET)
                     if f.endswith('.csv') and os.path.isfile(os.path.join(FOLDER_BOURGET, f))]

    all_files = files_leman + files_bourget
    print(f"Found {len(all_files)} files to process.")
    
    if not all_files:
        print("No CSV files found in the specified data folders.")
        return

    for file_to_process in all_files:
        process_file_refactored(file_to_process, OUTPUT_DIR)

if __name__ == "__main__":
    main()
