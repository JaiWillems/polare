

from stroke.interpolate._interpolant_utils import (
    _get_degree,
    _get_number_segments,
    _get_segments,
    _linear_power
)
from unittest import TestCase
import numpy as np


class TestGetDegree(TestCase):

    def test_get_degree(self):
        
        n = 10

        d = 8
        m_val = 8
        m_test = _get_degree(n, d)
        self.assertEqual(m_val, m_test)

        d = 9
        m_val = 9
        m_test = _get_degree(n, d)
        self.assertEqual(m_val, m_test)

        d = 10
        m_val = 9
        m_test = _get_degree(n, d)
        self.assertEqual(m_val, m_test)

        d = 11
        m_val = 9
        m_test = _get_degree(n, d)
        self.assertEqual(m_val, m_test)


class TestGetNumberSegments(TestCase):

    def test_get_number_segments(self):
        
        n = 10

        d = 1
        s_val = 9
        s_test = _get_number_segments(n, d)
        self.assertEqual(s_val, s_test)

        d = 2
        s_val = 5
        s_test = _get_number_segments(n, d)
        self.assertEqual(s_val, s_test)

        d = 3
        s_val = 3
        s_test = _get_number_segments(n, d)
        self.assertEqual(s_val, s_test)

        d = 4
        s_val = 3
        s_test = _get_number_segments(n, d)
        self.assertEqual(s_val, s_test)

        d = 5
        s_val = 2
        s_test = _get_number_segments(n, d)
        self.assertEqual(s_val, s_test)

        d = 6
        s_val = 2
        s_test = _get_number_segments(n, d)
        self.assertEqual(s_val, s_test)

        d = 7
        s_val = 2
        s_test = _get_number_segments(n, d)
        self.assertEqual(s_val, s_test)

        d = 8
        s_val = 2
        s_test = _get_number_segments(n, d)
        self.assertEqual(s_val, s_test)

        d = 9
        s_val = 1
        s_test = _get_number_segments(n, d)
        self.assertEqual(s_val, s_test)

        d = 10
        s_val = 1
        s_test = _get_number_segments(n, d)
        self.assertEqual(s_val, s_test)

        d = 11
        s_val = 1
        s_test = _get_number_segments(n, d)
        self.assertEqual(s_val, s_test)

        d = 12
        s_val = 1
        s_test = _get_number_segments(n, d)
        self.assertEqual(s_val, s_test)


class TestGetSegments(TestCase):

    def test_get_segments(self):
        
        n = 10
        x = np.linspace(1, 10, n)
        
        d = 1
        val_seg = np.array([[1, 2],
                            [2, 3],
                            [3, 4],
                            [4, 5],
                            [5, 6],
                            [6, 7],
                            [7, 8],
                            [8, 9],
                            [9, 10]])
        test_seg = _get_segments(x, d)
        self.assertTrue(np.array_equal(val_seg, test_seg))

        d = 2
        val_seg = np.array([[1, 3],
                            [3, 5],
                            [5, 7],
                            [7, 9],
                            [8, 10]])
        test_seg = _get_segments(x, d)
        self.assertTrue(np.array_equal(val_seg, test_seg))

        d = 3
        val_seg = np.array([[1, 4],
                            [4, 7],
                            [7, 10]])
        test_seg = _get_segments(x, d)
        self.assertTrue(np.array_equal(val_seg, test_seg))

        d = 4
        val_seg = np.array([[1, 5],
                            [5, 9],
                            [6, 10]])
        test_seg = _get_segments(x, d)
        self.assertTrue(np.array_equal(val_seg, test_seg))

        d = 5
        val_seg = np.array([[1, 6],
                            [5, 10]])
        test_seg = _get_segments(x, d)
        self.assertTrue(np.array_equal(val_seg, test_seg))

        d = 6
        val_seg = np.array([[1, 7],
                            [4, 10]])
        test_seg = _get_segments(x, d)
        self.assertTrue(np.array_equal(val_seg, test_seg))

        d = 7
        val_seg = np.array([[1, 8],
                            [3, 10]])
        test_seg = _get_segments(x, d)
        self.assertTrue(np.array_equal(val_seg, test_seg))

        d = 8
        val_seg = np.array([[1, 9],
                            [2, 10]])
        test_seg = _get_segments(x, d)
        self.assertTrue(np.array_equal(val_seg, test_seg))

        d = 9
        val_seg = np.array([[1, 10]])
        test_seg = _get_segments(x, d)
        self.assertTrue(np.array_equal(val_seg, test_seg))

        d = 11
        val_seg = np.array([[1, 10]])
        test_seg = _get_segments(x, d)
        self.assertTrue(np.array_equal(val_seg, test_seg))

        d = 12
        val_seg = np.array([[1, 10]])
        test_seg = _get_segments(x, d)
        self.assertTrue(np.array_equal(val_seg, test_seg))


class TestLinearPower(TestCase):

    def test_linear_power(self):

        mat1 = np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]])
        mat2 = np.array([[1, 2, 9],
                         [1, 5, 36],
                         [1, 8, 81]])
        mat3 = np.array([[1, 1, 1],
                         [4, 5, 6],
                         [49, 64, 81]])
        
        mat2test = _linear_power(mat1)
        self.assertTrue(np.array_equal(mat2, mat2test))
