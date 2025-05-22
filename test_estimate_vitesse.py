import numpy as np
import pandas as pd
import unittest
# Add the parent directory to the Python path to allow importing the refactored script
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from estimate_vitesse_refactored import find_most_linear_segment, calculate_slopes_and_r2

class TestEstimateVitesse(unittest.TestCase):

    def test_find_most_linear_segment(self):
        """
        Test find_most_linear_segment with a clear linear segment.
        """
        # Data: a flat segment, then linear increase, then flat again
        time = np.arange(30)
        depth = np.concatenate([
            np.full(10, 1.0),  # Flat segment
            np.linspace(1.0, 11.0, 10),  # Linear increase (slope 1)
            np.full(10, 11.0)  # Flat segment
        ])

        # We expect the function to find the linear segment (indices 10 to 19)
        # Window size for R2 should be a fraction of this segment's length (10).
        # If R2_WINDOW_SIZE_FRACTION is 4, window_size = 10/4 = 2.5, rounded to 2 or 3.
        # Let's test the segment [10:20] directly.
        
        segment_time = time[10:20]
        segment_depth = depth[10:20]

        # Call with a window_size_fraction that ensures a small enough window
        # to accurately identify the linear part.
        # The function defaults to R2_WINDOW_SIZE_FRACTION = 4.
        # For a 10-point segment, this means a window of 10/4=2.5, so 2 or 3.
        start_idx, end_idx, r2 = find_most_linear_segment(segment_time, segment_depth)

        # Assertions:
        # The R² should be very close to 1.0 for a perfectly linear segment.
        self.assertAlmostEqual(r2, 1.0, places=5, msg="R² should be close to 1.0 for a linear segment.")

        # Calculate expected window_size. R2_WINDOW_SIZE_FRACTION is 4 by default.
        expected_window_size = max(2, round(len(segment_time) / 4))
        # find_most_linear_segment has internal logic for too small windows or segments
        if expected_window_size >= len(segment_time): # if window is as large as segment
             expected_window_size = len(segment_time) -1 # Adjust to ensure at least 2 points for polyfit
        if expected_window_size < 2: # if segment is too small
             expected_window_size = len(segment_time)
        
        # The length of the identified segment should be equal to the window_size used by the function.
        self.assertEqual(end_idx - start_idx, expected_window_size,
                         msg=f"Length of identified segment should be {expected_window_size}. segment_time len: {len(segment_time)}")
        
        # Test with data that is not perfectly linear but has a clear best segment
        time_noisy = np.arange(20)
        depth_noisy = np.concatenate([
            np.linspace(0, 5, 5), # linear
            np.array([5.1, 5.3, 4.9, 5.2, 5.0]), # noisy flat
            np.linspace(5, 0, 5) # linear
        ])
        # Expect the first linear part [0:5]
        # Corrected: pass time_noisy and depth_noisy
        segment_len_noisy = 5
        time_segment_noisy = time_noisy[:segment_len_noisy]
        depth_segment_noisy = depth_noisy[:segment_len_noisy]
        
        # Calculate expected window_size for this segment
        # R2_WINDOW_SIZE_FRACTION is 4 by default in the tested function
        expected_window_size_noisy = max(2, round(len(time_segment_noisy) / 4))
        if expected_window_size_noisy >= len(time_segment_noisy): # if window is as large as segment
            expected_window_size_noisy = len(time_segment_noisy) -1
        if expected_window_size_noisy < 2: # if segment is too small
             expected_window_size_noisy = len(time_segment_noisy)


        start_idx_n, end_idx_n, r2_n = find_most_linear_segment(time_segment_noisy, depth_segment_noisy)
        
        self.assertAlmostEqual(r2_n, 1.0, places=5, msg="R2 for initial linear part of noisy data.")
        # The start_idx_n should be 0 as the first part is perfectly linear
        self.assertEqual(start_idx_n, 0, msg="Start index for noisy data linear segment.")
        # The length of the found segment should be the calculated window_size
        self.assertEqual(end_idx_n - start_idx_n, expected_window_size_noisy, 
                         msg=f"Length of segment for noisy data should be {expected_window_size_noisy}.")


    def test_calculate_slopes_and_r2_v_shape(self):
        """
        Test calculate_slopes_and_r2 with a synthetic V-shape dataset.
        """
        # Time array
        time = np.linspace(0, 100, 101) # 101 points for easier indexing for smoothing

        # Depth data: V-shape (descent then ascent)
        # Descent from depth 100 to 0 over 50 seconds (slope -2, but depth increases so +2)
        # Ascent from depth 0 to 100 over 50 seconds (slope +2, but depth decreases so -2)
        depth = np.concatenate([
            np.linspace(0, 100, 51)[:-1],  # Descent (0 to 100), remove last point to avoid overlap
            np.linspace(100, 0, 51)   # Ascent (100 to 0)
        ])
        # Total 50 + 51 = 101 points, matches time.

        # Parameters for calculate_slopes_and_r2
        epsilon = 0.1  # Slope threshold
        # Smoothing window: needs to be odd, and less than segment length.
        # Segments are ~50 points. A window of 5 or 7 should be fine.
        smoothing_window_val = 5 # Must be odd

        results, smoothed_depth = calculate_slopes_and_r2(time, depth, epsilon, smoothing_window_val)

        # Assertions for descent (inc) phase
        self.assertIn('inc', results, "Descent ('inc') phase should be detected.")
        inc_data = results['inc']
        self.assertAlmostEqual(inc_data['slope'], 2.0, delta=0.2,
                               msg="Descent slope should be approximately 2.0 m/s.")
        self.assertTrue(inc_data['r2'] > 0.9, msg="R² for descent should be high.")
        # Check if the detected time range is reasonable (e.g., within the first half)
        self.assertTrue(inc_data['start_time'] < 50, "Descent start time is incorrect.")
        self.assertTrue(inc_data['end_time'] <= 50, "Descent end time is incorrect.")


        # Assertions for ascent (dec) phase
        self.assertIn('dec', results, "Ascent ('dec') phase should be detected.")
        dec_data = results['dec']
        self.assertAlmostEqual(dec_data['slope'], -2.0, delta=0.2,
                               msg="Ascent slope should be approximately -2.0 m/s.")
        self.assertTrue(dec_data['r2'] > 0.9, msg="R² for ascent should be high.")
        # Check if the detected time range is reasonable (e.g., within the second half)
        self.assertTrue(dec_data['start_time'] >= 50, "Ascent start time is incorrect.")
        self.assertTrue(dec_data['end_time'] > 50, "Ascent end time is incorrect.")

        # Test with a more complex shape: flat, descent, flat, ascent, flat
        time_complex = np.linspace(0, 200, 201)
        depth_complex = np.concatenate([
            np.full(40, 0.0),                # Flat start
            np.linspace(0, 50, 61)[1:],      # Descent (50m over 60s, slope ~0.83)
            np.full(40, 50.0),               # Flat middle
            np.linspace(50, 0, 61)[1:],      # Ascent (50m over 60s, slope ~-0.83)
            np.full(40, 0.0)                 # Flat end
        ])
        # Total points: 40 + 60 + 40 + 60 + 40 = 240. Time needs to match.
        time_complex = np.linspace(0, len(depth_complex)-1, len(depth_complex))


        results_complex, _ = calculate_slopes_and_r2(time_complex, depth_complex, epsilon=0.1, smoothing_window_val=11)

        self.assertIn('inc', results_complex, "Descent ('inc') in complex shape.")
        inc_complex = results_complex['inc']
        self.assertAlmostEqual(inc_complex['slope'], 50.0/60.0, delta=0.15, msg="Complex descent slope.")
        self.assertTrue(inc_complex['r2'] > 0.9, msg="Complex descent R².")
        # Descent should be roughly between t=40 and t=100. Smoothing window = 11 (extends 5 points)
        # Actual start: 40. Detected can be 40-5=35. Actual end: 99. Detected can be 99+5=104.
        self.assertTrue(35 <= inc_complex['start_time'] < 45, f"Inc start: {inc_complex['start_time']}") 
        self.assertTrue(94 <= inc_complex['end_time'] < 105, f"Inc end: {inc_complex['end_time']}")


        self.assertIn('dec', results_complex, "Ascent ('dec') in complex shape.")
        dec_complex = results_complex['dec']
        self.assertAlmostEqual(dec_complex['slope'], -50.0/60.0, delta=0.15, msg="Complex ascent slope.")
        self.assertTrue(dec_complex['r2'] > 0.9, msg="Complex ascent R².")
        # Ascent should be roughly between t=140 and t=200. Smoothing window = 11 (extends 5 points)
        # Actual start: 140. Detected can be 140-5=135. Actual end: 199. Detected can be 199+5=204.
        self.assertTrue(135 <= dec_complex['start_time'] < 145, f"Dec start: {dec_complex['start_time']}")
        self.assertTrue(194 <= dec_complex['end_time'] < 205, f"Dec end: {dec_complex['end_time']}")


if __name__ == '__main__':
    unittest.main()
