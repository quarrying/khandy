import unittest
import copy
import numpy as np
import khandy


class TestEqLenSequences(unittest.TestCase):
    def setUp(self):
        self.confs = np.arange(10) * 0.1
        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.data = khandy.EqLenSequences(
            confs=self.confs,
            class_names=self.class_names
        )

    def _assert_equal(self, a, b):
        self.assertEqual(len(a), len(b))
        self.assertTrue(np.array_equal(a.confs, b.confs))
        self.assertEqual(a.class_names, b.class_names)
        
    def test_getitem(self):
        self._assert_equal(self.data[0], khandy.EqLenSequences(
            confs=self.confs[0:1], class_names=self.class_names[0:1]))
        self._assert_equal(self.data[-1], khandy.EqLenSequences(
            confs=self.confs[-1:], class_names=self.class_names[-1:]))
        self._assert_equal(self.data[1:3], khandy.EqLenSequences(
            confs=self.confs[1:3], class_names=self.class_names[1:3]))
        self._assert_equal(self.data[[1, 2, 3]], khandy.EqLenSequences(
            confs=self.confs[[1, 2, 3]], class_names=self.class_names[1:4]))
        self._assert_equal(self.data[np.array([1, 2, 3])], khandy.EqLenSequences(
            confs=self.confs[np.array([1, 2, 3])], class_names=self.class_names[1:4]))
        
        self._assert_equal(self.data[[1, 2, 4]], khandy.EqLenSequences(
            confs=self.confs[[1, 2, 4]], class_names=['1', '2', '4']))
        

        mask = self.data.confs > 0.5
        self._assert_equal(self.data[mask], khandy.EqLenSequences(
            confs=self.confs[mask], class_names=['6', '7', '8', '9']))

    def _test_filter(self, data: khandy.EqLenSequences, indices, inplace=False):
        if inplace:
            stub = copy.deepcopy(data)
            filtered = stub.filter(indices, inplace=True)
            self._assert_equal(stub, self.data[indices])
            self._assert_equal(filtered, self.data[indices])
        else:
            stub = data
            filtered = stub.filter(indices, inplace=False)
            self._assert_equal(stub, self.data)
            self._assert_equal(filtered, self.data[indices])
        
    def test_filter(self):
        for inplace in [True, False]:
            self._test_filter(self.data, 0, inplace=inplace)
            self._test_filter(self.data, -1, inplace=inplace)
            self._test_filter(self.data, slice(1,3), inplace=inplace)
            self._test_filter(self.data, [1,2,3], inplace=inplace)
            self._test_filter(self.data, np.array([1, 2, 3]), inplace=inplace)
            self._test_filter(self.data, self.confs > 0.5, inplace=inplace)
        

if __name__ == '__main__':
    unittest.main()
