import unittest
from parameterized import parameterized
import numpy as np
import sax_py
from scipy.stats import norm
from pyspark import SparkContext

sc = SparkContext(appName="unittest")


class TestZNormalizeSeries(unittest.TestCase):

    @parameterized.expand([[np.float('Inf')], [-np.float('Inf')], [np.float('NaN')]])
    def test_error_case(self, test_value):
        with self.assertRaises(ValueError):
            sax_py.SaxSeries._z_normalize_series(np.array([1, test_value]))

    def test_one_element(self):
        self.assertEqual(0, sax_py.SaxSeries._z_normalize_series(np.array(1)))

    def test_usual_case1(self):
        np.testing.assert_almost_equal([1, -1], sax_py.SaxSeries._z_normalize_series(np.array([1, -1])))

    def test_usual_case2(self):
        input_series = np.array([13.0, 225.3, 93.1, -1221.03])
        output_series = input_series - input_series.mean()
        output_series = output_series / output_series.std()

        np.testing.assert_almost_equal(output_series, sax_py.SaxSeries._z_normalize_series(input_series))


class TestPaaSeries(unittest.TestCase):

    def test_one_element(self):
        sax_series = sax_py.SaxSeries(np.array([1.0]), 1, 1)
        np.testing.assert_almost_equal([1.0], sax_series._paa_series(np.array([1.0])))

    @parameterized.expand([
        [np.array([2, 1]), np.array([1, 3, -1, 3]), 2, 1],
        [np.array([-1, 1, 11/3]), np.array([-1, 1, 2, 4, 5]), 3, 1]
    ])
    def test_usual_cases(self, output_array, input_array, w, a):
        sax_series = sax_py.SaxSeries(input_array, w, a)
        np.testing.assert_almost_equal(output_array, sax_series._paa_series(input_array))


class TestGetStndrdNormalBoundries(unittest.TestCase):

    @parameterized.expand([[1, np.array([-np.float('Inf'), np.float('Inf')])]])
    def test_one_interval_cases(self, a, expected_output_ndarray):
        np.testing.assert_almost_equal(expected_output_ndarray, sax_py.SaxSeries._get_stndrd_normal_boundries(a))

    @parameterized.expand([
        [2, np.array([-np.float('Inf'), 0.0, np.float('Inf')])],
        [4, np.array([-np.float('Inf'), norm.ppf(1/4), 0.0, norm.ppf(3/4), np.float('Inf')])]
    ])
    def test_usual_cases(self, a, expected_output_ndarray):
        np.testing.assert_almost_equal(expected_output_ndarray, sax_py.SaxSeries._get_stndrd_normal_boundries(a))


class TestDiscretizeYvalues(unittest.TestCase):

    @parameterized.expand([
        [np.array([-1.0, 0.0, 1.0]), -1, 2, np.array([1, 0, 0])],
        [np.array([2.0, 0.2, -0.1, -3.0]), -1, 4, np.array([0, 1, 2, 3])]
    ])
    def test_usual_case(self, input_ndarray, w, a, expected_output_ndarray):
        sax_series = sax_py.SaxSeries(input_ndarray, w, a)
        np.testing.assert_almost_equal(expected_output_ndarray, sax_series._discretize_yvalues(input_ndarray))


class TestGetBoundariesAssociatedToValue(unittest.TestCase):

    @parameterized.expand([
        [np.array([0, 1, 1, 2, 3, 0]), np.array([-np.float('Inf'), -1.0, 0.0, 1.0, np.float('Inf')]),
         np.array([(1.0, np.float('Inf')), (0.0, 1.0), (0.0, 1.0), (-1.0, 0.0), (-np.float('Inf'), -1.0), (1.0, np.float('Inf'))])]
    ])
    def test_usual_case(self, sax_series_ndarray, boundaries_ndarray, expected_output_ndarray):
        np.testing.assert_almost_equal(expected_output_ndarray, sax_py.SaxSeries._get_boundaries_associated_to_value(sax_series_ndarray, boundaries_ndarray))


class TestCountSymbols(unittest.TestCase):

    def test_usual_case(self):
        sax_series = sax_py.SaxSeries(np.array([1.0, 2.0]), 1, 4)
        sax_series._sax_series = np.array([0, 1, 2, 1, 1, 2])
        np.testing.assert_almost_equal(np.array([1, 3, 2, 0]), sax_series.get_symbol_count())


class TestSaxSeries(unittest.TestCase):

    @parameterized.expand([
        [np.array([2.0, -2.0, 4.0, -2.0, 0.0, 0.0, 4.0, 2.0]), 2, 4, np.array([2, 1])]
    ])
    def test_usual_case(self, input_ndarray, w, a, expected_output_ndarray):
        sax_series = sax_py.SaxSeries(input_ndarray, w, a)
        np.testing. assert_almost_equal(expected_output_ndarray, sax_series.get_sax_series())

    @parameterized.expand([
        [np.array([2.0, -2.0, 4.0, -2.0, 0.0, 0.0, 4.0, 2.0]), 2, 4, np.array([(norm.ppf(1/4), 0.0), (0.0, norm.ppf(3/4))])]
    ])
    def test_usual_case_boundaries(self, input_ndarray, w, a, expected_output_ndarray):
        sax_series = sax_py.SaxSeries(input_ndarray, w, a)
        np.testing. assert_almost_equal(expected_output_ndarray, sax_series.get_sax_series(sax_format=sax_series.SaxFormat.INTERVALBOUNDRIES))

    @parameterized.expand([
        [np.array([2.0, -2.0, 4.0, -2.0, 0.0, 0.0, 4.0, 2.0]), 2, 4, np.array([0, 1, 1, 0])]
    ])
    def test_usual_case_get_symbol_count(self, input_ndarray, w, a, expected_output_ndarray):
        sax_series = sax_py.SaxSeries(input_ndarray, w, a)
        np.testing. assert_almost_equal(expected_output_ndarray, sax_series.get_symbol_count())


class TestSaxSeriesRDD(unittest.TestCase):

    @parameterized.expand([
        [sc.parallelize(zip(range(8), [2.0, -2.0, 4.0, -2.0, 0.0, 0.0, 4.0, 2.0])), 2, 4, sc.parallelize(zip(np.array(range(2))*4, [2, 1]))]
    ])
    def test_usual_case(self, input_rdd, w, a, expected_output_rdd):
        sax_series_rdd = sax_py.SaxSeriesRDD(input_rdd, w, a)
        np.testing.assert_almost_equal(expected_output_rdd.collect(), sax_series_rdd.get_sax_series().collect())

    @parameterized.expand([
        [sc.parallelize(zip(range(8), [2.0, -2.0, 4.0, -2.0, 0.0, 0.0, 4.0, 2.0])), 2, 4, [0, 1, 1, 0]]
    ])
    def test_usual_case_get_symbol_count(self, input_rdd, w, a, expected_output_ndarray):
        sax_series = sax_py.SaxSeriesRDD(input_rdd, w, a)
        np.testing. assert_almost_equal(expected_output_ndarray, sax_series.get_symbol_count())


if __name__ == "__main__":
    unittest.main()