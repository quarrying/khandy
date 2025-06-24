import time

import khandy
import unittest


class TestBenchmark(unittest.TestCase):
    def test_benchmark_function(self):
        def sample_function(x):
            time.sleep(0.1) 
            return x * 2

        output, stats = khandy.benchmark(sample_function, args=(5,), num_repeats=5, num_repeats_burn_in=1)
        self.assertEqual(output, 10)  # 5 * 2
        self.assertIn('mean', stats)
        self.assertIn('stddev', stats)
        self.assertIn('max', stats)
        self.assertIn('min', stats)

    def test_benchmark_with_no_repeats(self):
        with self.assertRaises(AssertionError):
            khandy.benchmark(lambda x: x, num_repeats=-1)
        with self.assertRaises(AssertionError):
            khandy.benchmark(lambda x: x, num_repeats=3.5)
            

if __name__ == '__main__':
    unittest.main()
    
    