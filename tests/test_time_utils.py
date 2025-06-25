import time

import khandy
import unittest


class TestBenchmark(unittest.TestCase):
    def test_benchmark_function(self):
        def sample_function(x):
            time.sleep(0.1) 
            return x * 2

        output, stats = khandy.benchmark(
            sample_function, args=(5,), num_repeats=5, num_repeats_burn_in=1)
        self.assertEqual(output, 10)  # 5 * 2
        # Check the statistics
        self.assertGreater(stats.avg, 0.1)
        self.assertGreater(stats.std, 0)
        self.assertGreater(stats.min, 0)
        self.assertGreaterEqual(stats.max, stats.avg)
        self.assertEqual(stats.cnt, 5)
        self.assertAlmostEqual(stats.sum, stats.avg * stats.cnt)
        # Check that the aliases work correctly
        self.assertAlmostEqual(stats.mean, stats.avg)
        self.assertAlmostEqual(stats.stddev, stats.std)
        self.assertAlmostEqual(stats.total, stats.sum)
        self.assertEqual(stats.num_repeats, stats.cnt)
        
    def test_benchmark_with_no_repeats(self):
        with self.assertRaises(AssertionError):
            khandy.benchmark(lambda x: x, num_repeats=-1)
        with self.assertRaises(AssertionError):
            khandy.benchmark(lambda x: x, num_repeats=3.5)
            
    def test_benchmark_with_kwargs(self):
        def sample_function(x, multiplier=1):
            time.sleep(0.1)
            return x * multiplier

        output, stats = khandy.benchmark(
            sample_function, args=(5,), kwargs={'multiplier': 3}, num_repeats=3)
        self.assertEqual(output, 15)
        
if __name__ == '__main__':
    unittest.main()
    
    