import pandas as pd
import matplotlib.pyplot as plt
import os

# Define file paths
output_dir = "output"
leman_results_path = os.path.join(output_dir, "combined_results_Leman.csv")
bourget_results_path = os.path.join(output_dir, "combined_results_Bourget.csv")

# Read the data
try:
    leman_data = pd.read_csv(leman_results_path)
    bourget_data = pd.read_csv(bourget_results_path)
except FileNotFoundError:
    print(f"Error: One or both CSV files not found. Searched in {output_dir}")
    exit()

# Add a 'Lake' column to distinguish the data
leman_data['Lake'] = 'Leman'
bourget_data['Lake'] = 'Bourget'

# Combine the data
combined_data = pd.concat([leman_data, bourget_data])

# Create boxplots
fig, axes = plt.subplots(1, 2, figsize=(14, 7)) # Adjusted figsize for potentially more text

# Boxplot for inc_slope
combined_data.boxplot(column='inc_slope', by='Lake', ax=axes[0])
axes[0].set_title('Boxplot vitesse de descente') # Updated title
axes[0].set_ylabel('Vitesse de descente (m/s)') # Updated y-axis label
axes[0].set_xlabel('') # Remove default 'Lake' xlabel from boxplot

# Boxplot for dec_slope
combined_data.boxplot(column='dec_slope', by='Lake', ax=axes[1])
axes[1].set_title('Boxplot vitesse de remontée') # Updated title
axes[1].set_ylabel('Vitesse de remontée (m/s)') # Updated y-axis label
axes[1].set_xlabel('') # Remove default 'Lake' xlabel from boxplot

plt.suptitle('Comparison of Slopes for Leman and Bourget Lakes')
# plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle - will be called later

# Calculate and format summary statistics
def get_summary_text(df, column_name, lake_name):
    data = df[df['Lake'] == lake_name][column_name]
    median_val = data.median()
    min_val = data.min()
    max_val = data.max()
    # Added m/s unit to the summary text
    return f"Summary for {column_name} in {lake_name}:\n  Median: {median_val:.2f} m/s\n  Min: {min_val:.2f} m/s\n  Max: {max_val:.2f} m/s\n{'-'*30}\n"

summary_text = ""
summary_text += get_summary_text(combined_data, 'inc_slope', 'Leman')
summary_text += get_summary_text(combined_data, 'inc_slope', 'Bourget')
summary_text += get_summary_text(combined_data, 'dec_slope', 'Leman')
summary_text += get_summary_text(combined_data, 'dec_slope', 'Bourget')

# Add summary text to the figure
# Adjust x, y, and ha/va as needed, also fontsize
fig.text(0.99, 0.5, summary_text, transform=fig.transFigure, ha="right", va="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5))

# Adjust layout to make space for suptitle and text
# The rect parameter in tight_layout is [left, bottom, right, top] in figure coordinates.
# We reduce the 'right' to make space for the text added with fig.text on the right side.
plt.tight_layout(rect=[0, 0, 0.75, 0.96]) # Reduced right margin

plot_path = os.path.join(output_dir, 'slopes_boxplot_with_summary.png')
plt.savefig(plot_path)
print(f"Boxplot with summary saved to {plot_path}")

# Print summary to console as well
print("\n" + summary_text.replace("\n", "\n"))

plt.show() # Show plot at the end
