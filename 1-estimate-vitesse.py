import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# --- Configuration ---
OUTPUT_DIR = "output"
FOLDER_LEMAN = "data/Sonde_Profondeur-Leman/"
FOLDER_BOURGET = "data/Sonde_Profondeur-Bourget/"

# --- Ensure output directory exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_most_linear_segment(time, depth, window_size=100):
    """Find the most linear segment using sliding window and R² metric"""
    best_r2 = -1
    best_start = 0
    best_end = 0
    
    for i in range(len(time) - window_size):
        window_time = time[i:i+window_size]
        window_depth = depth[i:i+window_size]
        
        # Fit linear regression
        coef = np.polyfit(window_time, window_depth, 1)
        pred = np.polyval(coef, window_time)
        
        # Calculate R²
        r2 = 1 - (np.sum((window_depth - pred)**2) / 
                  np.sum((window_depth - np.mean(window_depth))**2))
        
        if r2 > best_r2:
            best_r2 = r2
            best_start = i
            best_end = i + window_size
    
    return best_start, best_end, best_r2

def detect_last_slopes(time, depth, *, epsilon=1e-3, smooth=True, window=51):
    """
    Return the final up‑slope (inc) and the following down‑slope (dec).
    """
    time   = np.asarray(time,  dtype=float)
    depth  = np.asarray(depth, dtype=float)

    if smooth:
        kernel      = np.ones(window) / window
        depth_sm    = np.convolve(depth, kernel, mode='same')
    else:
        depth_sm    = depth.copy()

    deriv        = np.gradient(depth_sm, time)
    pos_mask     = deriv >  epsilon
    neg_mask     = deriv < -epsilon

    def runs(mask):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return np.empty((0, 2), dtype=int)
        splits  = np.where(np.diff(idx) != 1)[0] + 1
        starts  = np.insert(idx[splits], 0, idx[0])
        ends    = np.append(idx[splits-1], idx[-1])
        return np.vstack((starts, ends)).T

    pos_runs = runs(pos_mask)
    neg_runs = runs(neg_mask)
    pos_runs_len = np.array([np.diff(run) for run in pos_runs])
    max_pos_run_idx = np.argmax(pos_runs_len)
    inc_run   = pos_runs[max_pos_run_idx] if pos_runs.size else None
    max_neg_run_idx = np.where(neg_runs > inc_run[1])
    dec_run   = neg_runs[max_neg_run_idx] if neg_runs.size else None

    result = {}
    if inc_run is not None:
        i0, i1 = inc_run
        inc_time = time[i0-window:i1+1+window]
        inc_depth = depth_sm[i0-window:i1+1+window]
         
         # Find most linear segment
        windowr2 = round(len(inc_time)/4)
        best_start, best_end, inc_r2 = find_most_linear_segment(inc_time, inc_depth, window_size=windowr2)
        inc_time = inc_time[best_start:best_end]
        inc_depth = inc_depth[best_start:best_end]
    
        inc_coef = np.polyfit(inc_time, inc_depth, 1)
        inc_speed = inc_coef[0]
        
        
        result['inc'] = dict(
            start        = float(time[i0]),
            end          = float(time[i1]),
            depth_start  = float(depth_sm[i0]),
            depth_end    = float(depth_sm[i1]),
            slope        = inc_speed
        )
    if dec_run is not None:
        j0, j1 = dec_run
        dec_time = time[j0-window:j1+1+window]
        dec_depth = depth_sm[j0-window:j1+1+window]
        
        # Find most linear segment
        windowr2 = round(len(dec_time)/4)
        best_start, best_end, dec_r2 = find_most_linear_segment(dec_time, dec_depth, window_size=windowr2)
        dec_time = dec_time[best_start:best_end]
        dec_depth = dec_depth[best_start:best_end]
        
        dec_coef = np.polyfit(dec_time, dec_depth, 1)
        dec_speed = dec_coef[0]
        result['dec'] = dict(
            start        = float(time[j0]),
            end          = float(time[j1]),
            depth_start  = float(depth_sm[j0]),
            depth_end    = float(depth_sm[j1]),
            slope        = dec_speed
        )
    return result, depth_sm

def find_header_line(file_path, pattern="Date and Time;Seconds     ;Depth (m)                               ;"):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for idx, raw in enumerate(f):
            line = raw.rstrip('\n')
            if pattern in line:
                return idx, line
    raise ValueError(f"Header pattern not found: {pattern!r}")

def get_lake_name(file_path):
    if "Sonde_Profondeur-Leman" in file_path:
        return "Leman"
    elif "Sonde_Profondeur-Bourget" in file_path:
        return "Bourget"
    else:
        return "Unknown"

def process_file(filetoread):
    idx, header = find_header_line(filetoread)
    data = pd.read_csv(filetoread, sep=';', skiprows=idx, encoding='utf-8', error_bad_lines=False)
    data = data.drop(data.columns[3], axis=1)
    data.columns = ['Date', 'Time', 'Depth']

    time  = data['Time'].values
    depth = data['Depth'].values
    res, depth_sm = detect_last_slopes(time, depth, epsilon=0.05, window=201)

    res['file'] = filetoread
    res['sampling_date'] = data['Date'].values[0]

    lake_name = get_lake_name(filetoread)
    date_str = str(res['sampling_date']).replace("/", "-").replace(" ", "_")

    # Save result as CSV
    output_csv = os.path.join(OUTPUT_DIR, f"result_{lake_name}_{date_str}.csv")
    df = pd.json_normalize([res], sep='_')
    df.to_csv(output_csv, index=False)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(time, depth, label="raw")
    plt.plot(time, depth_sm, linewidth=1.5, label="smoothed")
    plt.axvspan(res['inc']['start'], res['inc']['end'], alpha=0.2)
    plt.axvspan(res['dec']['start'], res['dec']['end'], alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel("Depth (m)")

    # Annotate in the middle of y
    ymin, ymax = plt.ylim()
    y_middle = (ymin + ymax) / 2

    plt.plot([res['inc']['start'], res['inc']['end']], [res['inc']['depth_start'], res['inc']['depth_end']], 'r', label="inc slope")
    plt.plot([res['dec']['start'], res['dec']['end']], [res['dec']['depth_start'], res['dec']['depth_end']], 'g', label="dec slope")
    plt.annotate(
        f"Descente: {res['inc']['slope']:.2f} m/s",
        xy=(res['inc']['start'], y_middle),
        xytext=(res['inc']['start'], y_middle),
        ha='left', va='center',
        arrowprops=dict(arrowstyle="->")
    )
    plt.annotate(
        f"Remonte: {res['dec']['slope']:.2f} m/s",
        xy=(res['dec']['start'], y_middle),
        xytext=(res['dec']['start'], y_middle),
        ha='left', va='center',
        arrowprops=dict(arrowstyle="->")
    )
    plt.title(f"Detected final slopes\nFile: {res['file']}\nDate: {res['sampling_date']}")
    plt.legend()

    output_plot = os.path.join(OUTPUT_DIR, f"plot_{lake_name}_{date_str}.png")
    plt.tight_layout()
    plt.savefig(output_plot, bbox_inches='tight', pad_inches=0.5)
    plt.close()  # Close the figure to avoid display and memory issues

def main():
    # Process all files in Leman and Bourget folders
    files_leman = [os.path.join(FOLDER_LEMAN, f) for f in os.listdir(FOLDER_LEMAN) if f.endswith('.csv')]
    files_bourget = [os.path.join(FOLDER_BOURGET, f) for f in os.listdir(FOLDER_BOURGET) if f.endswith('.csv')]
    print("Leman:", files_leman)
    print("Bourget:", files_bourget)

    for filetoread in files_leman + files_bourget:
        print(f"Processing {filetoread}")
        try:
            process_file(filetoread)
        except Exception as e:
            print(f"Error processing {filetoread}: {e}")

if __name__ == "__main__":
    main()