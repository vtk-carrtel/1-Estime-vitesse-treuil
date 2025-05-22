import unittest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, mock_open

# Add the parent directory to the Python path to allow importing the refactored script
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Functions to test from the refactored script
# Assuming the refactored script is named 'boxplot_refactored.py' for import
from boxplot_refactored import (
    load_and_prepare_data,
    generate_summary_text_for_plot,
    create_slope_comparison_boxplots, # For potential future tests, not tested directly now for plotting
    OUTPUT_DIR, # Import if used directly in tests, e.g. for constructing paths
    LEMAN_RESULTS_FILENAME,
    BOURGET_RESULTS_FILENAME
)

class TestBoxplot(unittest.TestCase):

    def test_generate_summary_text_for_plot(self):
        """
        Test the generate_summary_text_for_plot function for correct formatting and statistics.
        """
        data = {
            'inc_slope': [1.0, 1.5, 2.0, 10.0, 10.5, 11.0], # Leman: 1.0, 1.5, 2.0; Bourget: 10.0, 10.5, 11.0
            'dec_slope': [0.5, 0.7, 0.9, 12.0, 12.5, 13.0], # Leman: 0.5, 0.7, 0.9; Bourget: 12.0, 12.5, 13.0
            'Lake': ['Leman', 'Leman', 'Leman', 'Bourget', 'Bourget', 'Bourget']
        }
        df = pd.DataFrame(data)

        summary_text = generate_summary_text_for_plot(df)

        # Expected statistics for Leman inc_slope: Median=1.50, Min=1.00, Max=2.00
        self.assertIn("Summary for inc_slope in Leman:", summary_text)
        self.assertIn("Median: 1.50 m/s", summary_text)
        self.assertIn("Min: 1.00 m/s", summary_text)
        self.assertIn("Max: 2.00 m/s", summary_text)

        # Expected statistics for Bourget inc_slope: Median=10.50, Min=10.00, Max=11.00
        self.assertIn("Summary for inc_slope in Bourget:", summary_text)
        self.assertIn("Median: 10.50 m/s", summary_text)
        self.assertIn("Min: 10.00 m/s", summary_text)
        self.assertIn("Max: 11.00 m/s", summary_text)

        # Expected statistics for Leman dec_slope: Median=0.70, Min=0.50, Max=0.90
        self.assertIn("Summary for dec_slope in Leman:", summary_text)
        self.assertIn("Median: 0.70 m/s", summary_text)
        self.assertIn("Min: 0.50 m/s", summary_text)
        self.assertIn("Max: 0.90 m/s", summary_text)
        
        # Expected statistics for Bourget dec_slope: Median=12.50, Min=12.00, Max=13.00
        self.assertIn("Summary for dec_slope in Bourget:", summary_text)
        self.assertIn("Median: 12.50 m/s", summary_text)
        self.assertIn("Min: 12.00 m/s", summary_text)
        self.assertIn("Max: 13.00 m/s", summary_text)

        # Test with missing column
        df_missing_col = pd.DataFrame({'Lake': ['Leman'], 'inc_slope': [1]})
        summary_missing = generate_summary_text_for_plot(df_missing_col)
        self.assertIn("Column 'dec_slope' not found for Leman.", summary_missing)

        # Test with empty dataframe
        summary_empty = generate_summary_text_for_plot(pd.DataFrame())
        self.assertEqual(summary_empty, "No data available to generate summary.")


    @patch('pandas.read_csv')
    def test_load_and_prepare_data(self, mock_read_csv):
        """
        Test the load_and_prepare_data function for correct DataFrame creation and structure.
        """
        # Sample data for mocked CSV reads
        leman_sample_data = {'inc_slope': [1.0, 1.5], 'dec_slope': [0.5, 0.7]}
        bourget_sample_data = {'inc_slope': [10.0, 10.5], 'dec_slope': [12.0, 12.5]}

        # Configure mock_read_csv to return different DataFrames based on input path
        def read_csv_side_effect(file_path):
            if LEMAN_RESULTS_FILENAME in file_path:
                return pd.DataFrame(leman_sample_data)
            elif BOURGET_RESULTS_FILENAME in file_path:
                return pd.DataFrame(bourget_sample_data)
            else:
                raise FileNotFoundError(f"Unexpected file path: {file_path}")

        mock_read_csv.side_effect = read_csv_side_effect

        # Call the function to test
        combined_df = load_and_prepare_data(OUTPUT_DIR, LEMAN_RESULTS_FILENAME, BOURGET_RESULTS_FILENAME)

        # Assertions
        self.assertFalse(combined_df.empty)
        self.assertEqual(len(combined_df), 4) # 2 rows from Leman + 2 rows from Bourget
        self.assertIn('Lake', combined_df.columns)
        
        # Check data integrity and 'Lake' column assignment
        self.assertEqual(len(combined_df[combined_df['Lake'] == 'Leman']), 2)
        self.assertEqual(len(combined_df[combined_df['Lake'] == 'Bourget']), 2)
        
        pd.testing.assert_series_equal(
            combined_df[combined_df['Lake'] == 'Leman']['inc_slope'].reset_index(drop=True),
            pd.Series(leman_sample_data['inc_slope'], name='inc_slope')
        )
        pd.testing.assert_series_equal(
            combined_df[combined_df['Lake'] == 'Bourget']['inc_slope'].reset_index(drop=True),
            pd.Series(bourget_sample_data['inc_slope'], name='inc_slope')
        )

        # Test FileNotFoundError
        mock_read_csv.side_effect = FileNotFoundError("Mocked FileNotFoundError")
        error_df_fnf = load_and_prepare_data(OUTPUT_DIR, "non_existent_Leman.csv", "non_existent_Bourget.csv")
        self.assertTrue(error_df_fnf.empty, "Expected empty DataFrame on FileNotFoundError")

        # Test EmptyDataError
        mock_read_csv.side_effect = pd.errors.EmptyDataError("Mocked EmptyDataError")
        error_df_ede = load_and_prepare_data(OUTPUT_DIR, LEMAN_RESULTS_FILENAME, BOURGET_RESULTS_FILENAME)
        self.assertTrue(error_df_ede.empty, "Expected empty DataFrame on EmptyDataError")
        
        # Test general Exception
        mock_read_csv.side_effect = Exception("Mocked general exception")
        error_df_exc = load_and_prepare_data(OUTPUT_DIR, LEMAN_RESULTS_FILENAME, BOURGET_RESULTS_FILENAME)
        self.assertTrue(error_df_exc.empty, "Expected empty DataFrame on general Exception")


if __name__ == '__main__':
    # This allows running the tests directly if the script is executed
    # Requires matplotlib to be available if create_slope_comparison_boxplots is called by main()
    # For now, we are testing data functions, so it should be fine.
    # If main() in boxplot_refactored calls plt.show(), tests might hang if not handled.
    # The current main() in refactored script does not call plt.show() by default.
    unittest.main()
