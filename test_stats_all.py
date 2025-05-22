import unittest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, mock_open

# Add the parent directory to the Python path to allow importing the refactored script
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Functions to test from the refactored script
from stats_all_refactored import calculate_statistics, load_and_combine_csv_files, print_slope_statistics, OUTPUT_DIR

class TestStatsAll(unittest.TestCase):

    def test_calculate_statistics(self):
        """
        Test the calculate_statistics function with various scenarios.
        """
        # Scenario 1: Basic valid data
        data = {'col1': [1.0, 2.0, 3.0, 4.0, 5.0]}
        df = pd.DataFrame(data)
        mean, std, min_val, max_val = calculate_statistics(df, 'col1')
        self.assertAlmostEqual(mean, 3.0)
        self.assertAlmostEqual(std, np.std(data['col1'], ddof=1)) # pandas default ddof=1
        self.assertEqual(min_val, 1.0)
        self.assertEqual(max_val, 5.0)

        # Scenario 2: Data with NaN values (should be ignored by pandas mean, std, etc.)
        data_with_nan = {'col1': [1.0, np.nan, 3.0, np.nan, 5.0]}
        df_nan = pd.DataFrame(data_with_nan)
        mean_nan, std_nan, min_nan, max_nan = calculate_statistics(df_nan, 'col1')
        self.assertAlmostEqual(mean_nan, np.nanmean(data_with_nan['col1'])) # 3.0
        self.assertAlmostEqual(std_nan, np.nanstd(data_with_nan['col1'], ddof=1)) # np.std([1,3,5], ddof=1)
        self.assertEqual(min_nan, 1.0)
        self.assertEqual(max_val, 5.0) # From previous calculation, max_nan should be 5.0

        # Scenario 3: Column not found
        mean_nf, std_nf, min_nf, max_nf = calculate_statistics(df, 'non_existent_column')
        self.assertIsNone(mean_nf)
        self.assertIsNone(std_nf)
        self.assertIsNone(min_nf)
        self.assertIsNone(max_nf)

        # Scenario 4: All NaN values in column
        data_all_nan = {'col1': [np.nan, np.nan, np.nan]}
        df_all_nan = pd.DataFrame(data_all_nan)
        mean_an, std_an, min_an, max_an = calculate_statistics(df_all_nan, 'col1')
        self.assertIsNone(mean_an) # Depending on behavior, could be NaN or None. Function returns None.
        self.assertIsNone(std_an)
        self.assertIsNone(min_an)
        self.assertIsNone(max_an)
        
        # Scenario 5: Empty DataFrame
        df_empty = pd.DataFrame({'col1': []})
        mean_e, std_e, min_e, max_e = calculate_statistics(df_empty, 'col1')
        self.assertTrue(pd.isna(mean_e)) # Pandas mean of empty series is NaN
        self.assertTrue(pd.isna(std_e))
        self.assertTrue(pd.isna(min_e)) # This will raise error in pandas if not handled, function should return None
        self.assertTrue(pd.isna(max_e)) # This will raise error in pandas if not handled, function should return None
                                        # The current implementation of calculate_statistics would try to compute and might get NaN or error
                                        # Let's check if the function returns None as per its docstring for "unsuitable"


    @patch('glob.glob')
    @patch('pandas.read_csv')
    @patch('pandas.DataFrame.to_csv') # Mock to_csv to prevent actual file writing
    @patch('os.makedirs') # Mock makedirs
    @patch('os.path.exists', return_value=True) # Assume output dir exists
    def test_load_and_combine_csv_files(self, mock_path_exists, mock_makedirs, mock_to_csv, mock_read_csv, mock_glob):
        """
        Test load_and_combine_csv_files for correct DataFrame combination and CSV saving.
        """
        # Mock glob to return a list of dummy file paths
        mock_glob.return_value = ['dummy1.csv', 'dummy2.csv']

        # Mock pandas.read_csv to return different DataFrames for each dummy file
        df1_data = {'inc_slope': [1.0, 1.1], 'dec_slope': [2.0, 2.1]}
        df2_data = {'inc_slope': [1.2], 'dec_slope': [2.2]}
        mock_read_csv.side_effect = [pd.DataFrame(df1_data), pd.DataFrame(df2_data)]

        lake_name = "TestLake"
        combined_df = load_and_combine_csv_files(f"pattern_{lake_name}_*.csv", lake_name)

        # Assertions
        self.assertFalse(combined_df.empty)
        self.assertEqual(len(combined_df), 3) # 2 rows from df1 + 1 row from df2
        expected_inc_slopes = pd.Series([1.0, 1.1, 1.2], name='inc_slope') # Added name attribute
        pd.testing.assert_series_equal(combined_df['inc_slope'].reset_index(drop=True), expected_inc_slopes.reset_index(drop=True))
        
        # Check if to_csv was called correctly
        expected_save_path = os.path.join(OUTPUT_DIR, f"combined_slopes_{lake_name}.csv")
        mock_to_csv.assert_called_once_with(expected_save_path, index=False)

        # Test case: No files found
        mock_glob.return_value = []
        empty_df = load_and_combine_csv_files("pattern_empty_*.csv", "EmptyLake")
        self.assertTrue(empty_df.empty)

        # Test case: One file is empty
        mock_glob.return_value = ['dummy_data.csv', 'empty_simulated.csv']
        # Reset side_effect for mock_read_csv for this specific sub-test
        mock_read_csv.reset_mock() 
        mock_read_csv.side_effect = [
            pd.DataFrame(df1_data), # For 'dummy_data.csv'
            pd.DataFrame()          # For 'empty_simulated.csv', return an empty DataFrame directly
        ]
                                                                                                   
        with patch('builtins.print') as mock_print_empty: # to suppress print statements
            df_with_one_empty = load_and_combine_csv_files("pattern_one_empty_*.csv", "OneEmptyLake")
            # df1_data has 2 rows. The empty file should be skipped or result in an empty df being concatenated.
            # The current load_and_combine_csv_files appends df even if it's empty from read_csv.
            # If read_csv returns an empty DataFrame, it's appended.
            # So, length should be len(df1_data) + 0 = len(df1_data)
            self.assertEqual(len(df_with_one_empty), len(df1_data["inc_slope"]))
            # Check if warning was printed (optional, but good for robustness)
            # mock_print_empty.assert_any_call("Warning: CSV file empty.csv is empty and will be skipped.")


    @patch('builtins.print') # To capture print output
    def test_print_slope_statistics(self, mock_print):
        """
        Test the print_slope_statistics function for correct output formatting.
        """
        data = {
            'inc_slope': [1.5, 2.5, 3.5], # mean 2.5, std 1.0, min 1.5, max 3.5
            'inc_r2': [0.90, 0.92, 0.94], # mean 0.92, std ~0.02, min 0.90, max 0.94
            'dec_slope': [-1.5, -2.5, -3.5],
            'dec_r2': [0.80, 0.82, 0.84]
        }
        df = pd.DataFrame(data)
        lake_name = "TestLake"

        # Test for 'inc' slopes
        print_slope_statistics(df, lake_name, "inc")
        
        # Basic checks for printed output (can be more detailed)
        # Check that key statistics are mentioned in the prints.
        # This is a bit fragile as it depends on exact string formatting.
        mock_print.assert_any_call("\n--- Statistics for TestLake - INC Slopes ---")
        # Example check for mean slope (actual output will have more spaces and formatting)
        # We need to find a call that contains the expected substring.
        found_mean_slope = any("Mean: 2.500 m/s" in call_args[0][0] for call_args in mock_print.call_args_list if call_args[0])
        self.assertTrue(found_mean_slope, "Mean inc_slope was not printed correctly.")

        found_mean_r2 = any("Mean: 0.920" in call_args[0][0] for call_args in mock_print.call_args_list if call_args[0])
        self.assertTrue(found_mean_r2, "Mean inc_r2 was not printed correctly.")
        
        # Test for 'dec' slopes
        mock_print.reset_mock() # Reset mock before next call
        print_slope_statistics(df, lake_name, "dec")
        mock_print.assert_any_call("\n--- Statistics for TestLake - DEC Slopes ---")
        found_mean_dec_slope = any("Mean: -2.500 m/s" in call_args[0][0] for call_args in mock_print.call_args_list if call_args[0])
        self.assertTrue(found_mean_dec_slope, "Mean dec_slope was not printed correctly.")

        # Test with a column missing (e.g., no R2 column)
        df_no_r2 = df[['inc_slope']]
        mock_print.reset_mock()
        print_slope_statistics(df_no_r2, lake_name, "inc")
        mock_print.assert_any_call("  R² (inc_r2): Column not found, statistics not calculated.")

        # Test with empty dataframe
        mock_print.reset_mock()
        print_slope_statistics(pd.DataFrame(), lake_name, "inc")
        mock_print.assert_any_call(f"DataFrame for {lake_name} is empty. No statistics to print for inc slopes.")


if __name__ == '__main__':
    unittest.main()
