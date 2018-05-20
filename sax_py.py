import numpy as np
from scipy.stats import norm
import bisect

from enum import Enum

# Visualization
from bokeh.plotting import figure
from bokeh.layouts import row

# Introducing Abstract Base Classes
from abc import ABC, abstractmethod


class AbstractSaxSeries(ABC):
    """
    This is an abstract class. SaxSeries and SaxSeriesRDD inhert from this class.
    Further this class provides the _get_stndrd_normal_boundries(a) method.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_sax_series(self):
        pass

    @abstractmethod
    def get_symbol_count(self):
        pass

    @staticmethod
    def _get_stndrd_normal_boundries(a):
        boundaries = []
        for index in range(0, a + 1):
            boundaries.append(norm.ppf(index/a))

        return np.array(boundaries)


class SaxSeriesRDD(AbstractSaxSeries):
    """
    This class implements the sax conversion from a pyspark.RDD.rdd to a sax approximation.
    Further this class provides a method for counting the occurrences of the different sax symbols.
    """

    def __init__(self, series_rdd, w, a):
        """
        :param series_rdd: A pyspark.rdd.RDD, containing the time series which should be compressed.
        :param w: An int, the number of symbols.
        :param a: An int, the alphabet size. This has to be of the form 2^n.
        """

        self.series_rdd = series_rdd
        self.w = w
        self.a = a

        # Calculating the d the number of consecutive elements over which the average is taken.
        # Such that self.w or self.w + 1 symbols represent the time series.
        n = series_rdd.count()
        d = n // self.w
        # In case self.w does not divide n the last word will be based on more than d elements to guarantee that w
        # symbols represent the time series.
        self.d = d

        self._sax_rdd = None
        self._boundaries = None
        self._symbol_count = None

    def get_sax_series(self):
        """
        This function returns the sax series.
        :return: A pyspark.RDD.rdd, the sax approximation of the raw series.
        """
        if self._sax_rdd is None:  # Lazy initialization.
            self._calculate_sax_series()
        return self._sax_rdd

    def get_symbol_count(self):
        """
        :return: A list, containing the number of occurrences of each sax symbol. The indices correspond to the sax
        symbols.
        """
        if self._symbol_count is None:  # Lazy initialization.
            self._count_symbols()
        return self._symbol_count

    def _calculate_sax_series(self):
        """
        The sax series approximation is calculated.
        """
        mean_series = self.series_rdd.map(lambda x: x[1]).mean()
        stdev_series = self.series_rdd.map(lambda x: x[1]).stdev()

        z_normalized_series_rdd = self.series_rdd.map(lambda x: (x[0], (x[1] - mean_series) / stdev_series))

        aggregated_rdd = SaxSeriesRDD.sliding(z_normalized_series_rdd, self.d)

        paa_rdd =  aggregated_rdd.map(lambda x: (x[0], np.array(x[1]).mean()))

        boundaries = SaxSeriesRDD._get_stndrd_normal_boundries(self.a)
        self._boundaries = boundaries
        size_boundaries = boundaries.size
        self._sax_rdd = paa_rdd.map(lambda x: (x[0], size_boundaries - 1 - bisect.bisect_right(boundaries, x[1])))

    def _count_symbols(self):
        """
        Counting the number of occurrences.
        """
        if self._sax_rdd is None:
            self._calculate_sax_series()

        symbol_count = [0 for _ in range(self.a)]

        for ele in self._sax_rdd.map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x+y).collect():
            symbol_count[ele[0]] += ele[1]

        self._symbol_count = symbol_count

    @staticmethod
    def sliding(rdd, d):
        """
        :param rdd: A pyspark.RDD.rdd, containing the z-normalized series.
        :param d: An int, specifying the length of the aggregation window size.
        :return: An pyspark.RDD.rdd, the aggregated values.
        """
        assert d > 0

        def gen_window(ix, d):
            i, x = ix
            return [(i - offset, (i, x)) for offset in range(d)]

        return rdd.flatMap(lambda ix: gen_window(ix, d)) \
            .groupByKey().filter(lambda x: x[0] % d == 0) \
            .mapValues(lambda vals: [x for (i, x) in sorted(vals)]) \
            .sortByKey()


class SaxSeries(AbstractSaxSeries):
    """
    This class implements the sax conversion from a series to a sax approximation of the series.
    Further this class provides a method for counting the occurrences of the different sax symbols.
    """

    class SaxFormat(Enum):
        """
        This enum is used to specify the return value format of the get_sax_series() method.
        """
        INTVAL = 0  # The usual int values from 0 to a. Where 0 corresponds to the highest value and a to the lowest.
        BOOLVAL = 1  # Simply the transformation of the aboved mentioned integers into their binary representation.
        INTERVALBOUNDRIES = 2  # A tuple containing the lower and upper bound of the respective sax symbol.

    def __init__(self, series, w, a):
        """
        :param series: A numpy.ndarray, containing the time series which should be compressed.
        :param w: An int, the number of symbols.
        :param a: An int, the alphabet size. This has to be of the form 2^n.
        """

        self.series = series
        self.w = w
        self.a = a

        # Calculating the d the number of consecutive elements over which the average is taken.
        # Such that self.w symbols represent the time series.
        n = len(series)
        d = n // self.w
        # In case self.w does not divide n the last word will be based on more than d elements to guarantee that w
        # symbols represent the time series.
        self.d = d

        self._sax_series = None
        self._boundaries = None
        self._symbol_count = None

    def get_sax_series(self, sax_format=SaxFormat.INTVAL):
        """
        This function returns the sax series in the specified format.
        :param sax_format: SaxFormat, specifying the format of the return value.
        :return: The sax approximation of the raw series, the format is specified by the sax_format parameter.
        """
        if self._sax_series is None:  # Lazy initialization
            self._calculate_sax_series()

        if sax_format == SaxSeries.SaxFormat.INTVAL:
            return self._sax_series
        elif sax_format == SaxSeries.SaxFormat.BOOLVAL:
            return np.array(list(map(bin, self._sax_series)))
        else:
            return SaxSeries._get_boundaries_associated_to_value(self._sax_series, self.get_boundaries())

    def get_symbol_count(self):
        """
        :return: A list, containing the number of occurrences of each sax symbol. The indices correspond to the sax
        symbols.
        """
        if self._symbol_count is None:  # Lazy initialization
            self._count_symbols()
        return self._symbol_count

    def _count_symbols(self):
        """
        Counting the number of occurrences.
        """
        if self._sax_series is None:
            self._calculate_sax_series()
        symbol_count = [0 for _ in range(self.a)]
        for ele in self._sax_series:
            symbol_count[ele] += 1
        self._symbol_count = symbol_count

    @staticmethod
    def _get_boundaries_associated_to_value(sax_series_ndarray, boundaries_ndarray):
        """
        :param sax_series_ndarray: A numpy.ndarray, the sax series.
        :param boundaries_ndarray: A numpy.ndarray, the sax interval boundaries.
        :return: A list of tuples, each tuple contains the upper and the lower bound of the sax symbol associated to the
        current index.
        """
        return_list = []
        count_boundaries = boundaries_ndarray.size
        for index in range(sax_series_ndarray.size):
            current_index = count_boundaries - 1 - sax_series_ndarray[index]
            return_list.append((boundaries_ndarray[current_index - 1], boundaries_ndarray[current_index]))
        return return_list

    def get_boundaries(self):
        """
        :return: A numpy.ndarray, the boundaries of the sax symbol intervals.
        """
        if self._boundaries is None:  # Lazy initialization
            self._boundaries = SaxSeries._get_stndrd_normal_boundries(self.a)
        return self._boundaries

    def _calculate_sax_series(self):
        """
        The sax approximation of the series is calculated.
        """
        # 1.) Z-Normalization
        normalized_series = self._z_normalize_series(self.series)

        # 2.) Piecewise aggregate approximation of the normalized series.
        paa_series = self._paa_series(normalized_series)

        # 3.) The final sax_series
        self._sax_series = self._discretize_yvalues(paa_series)

    @staticmethod
    def _z_normalize_series(series):
        """
        This method z normalizes the series. That is to say the mean of the series is subtracted from each element
        and in addition each resulting element is divided the standard deviation.
        :param series: A numpy.ndarray, the raw series.
        :return: A numpy.ndarray, the z-normalized series.
        """
        mean = series.mean()
        if mean == np.float('Inf') or mean == -np.float('Inf') or np.isnan(mean):
            raise ValueError('The input series contains either a plus or minus Inf or a NaN.')
        series = series - mean
        if series.std() != 0:
            series = series / series.std()
        return series

    def _paa_series(self, series):
        """
        This method implements the piecewise aggregate approximation (paa) of the input series.
        :param series: A numpy.ndarray, the z-normalized series.
        :return: A numpy.ndarray, the paa of the input series.
        """
        return_list = []
        n = len(series)
        d = self.d

        for index_start_cur_window in range(self.w):
            if index_start_cur_window == self.w-1:
                return_list.append(series[index_start_cur_window*d: n].mean())
            else:
                return_list.append(series[index_start_cur_window*d: index_start_cur_window*d + d].mean())

        return np.array(return_list)

    def _discretize_yvalues(self, series):
        """
        This method ipmlements the transformation of the paa approximation into the sax approximation. That is to say
        the values of the time-wise aggregated series are discretized.
        :param series: A numpy.ndarray, the paa version of the series provided to the constructor of this class.
        :return: A numpy.ndarray, the sax approximation of the series.
        """
        return_list = []
        boundaries = self.get_boundaries()
        size_boundaries = boundaries.size

        for index in range(series.size):
            return_list.append(size_boundaries - 1 - bisect.bisect_right(boundaries, series[index]))

        return np.array(return_list)


def plot_series_vs_sax(series_ndarray, z_norm_ndarray, paa_ndarray, sax_interval_boundaries, general_boundaries, w, d):
    """
    Visualizing the SaxSeries transformations.

    :param series_ndarray:
    :param z_norm_ndarray:
    :param paa_ndarray:
    :param sax_interval_boundaries:
    :param general_boundaries:
    :param w:
    :param d:
    :return:
    """
    size_series = series_ndarray.size

    q = figure(title="Input Series", x_axis_label="Time", y_axis_label="Values", plot_width=350, plot_height=350)
    q.circle(range(size_series), series_ndarray, legend="Input Series")
    q.legend.background_fill_alpha = 0.0
    q.legend.location = "top_left"

    p = figure(title= "SAX Approximation", x_axis_label="Time", y_axis_label="Values", plot_width=600,
               plot_height=350)

    # Plotting the boundary lines
    general_boundaries = np.array(list(general_boundaries)[::-1])
    for index in range(1, len(general_boundaries) - 1):
        p.line([-0.5, size_series-1 + 0.5], [general_boundaries[index]] * 2, line_width=0.5, line_color='black',
               alpha=0.7, legend="Boundary Lines", muted_alpha=0.0)

    # Plotting the series values
    p.circle(range(size_series), z_norm_ndarray, legend="Z-Normalized Series", muted_alpha=0.0)

    # Plotting the piecewise aggregate approximations
    indices_paa = [index*d for index in range(w)]
    x0_paa = np.array(indices_paa) - 0.5
    x1_paa = indices_paa[1:]
    x1_paa.append(size_series)
    x1_paa = np.array(x1_paa) - 0.5
    p.segment(x0=x0_paa, y0=paa_ndarray, x1=x1_paa, y1=paa_ndarray, line_color="#2ca25f", legend="PAA Approximation",
              muted_alpha=0.0)

    # Plotting the SAX areas
    min_val = min(z_norm_ndarray.min(), general_boundaries[1:-1].min()) - 0.5
    max_val = max(z_norm_ndarray.max(), general_boundaries[1:-1].max()) + 0.5

    def inf_converter(value):
        if value == np.float('Inf'):
            return max_val
        elif value == -np.float('Inf'):
            return min_val
        else:
            return value
    bottom = [inf_converter(tuple[0]) for tuple in sax_interval_boundaries]
    x_coor = (np.array(x1_paa) + np.array(x0_paa)) / 2.0
    width = np.array(x1_paa) - np.array(x0_paa)
    top = [inf_converter(tuple[1]) for tuple in sax_interval_boundaries]
    p.vbar(x=x_coor, top=top, bottom=bottom, width=width, alpha=0.5, color="#99d8c9", legend="SAX Approximation",
           muted_alpha=0.0)

    p.legend.background_fill_alpha = 0.0
    p.legend.click_policy = "mute"
    p.legend.location = "top_left"

    return row(q, p)

