import unittest

suite = unittest.defaultTestLoader.discover(
    start_dir='./tests',
    pattern='test*.py',
    top_level_dir=None
)
# runner = unittest.TextTestRunner()
runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)