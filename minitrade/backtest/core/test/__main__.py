import sys
import unittest

suite = unittest.defaultTestLoader.discover('minitrade.backtest.core.test',
                                            pattern='_test*.py', top_level_dir='minitrade')
if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
