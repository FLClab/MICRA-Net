
import numpy

from sklearn import decomposition
from skimage import filters

class Combiner:
    def __init__(self):
        pass

    def flatten(self, ary, method):
        """
        Flattens the `numpy.ndarray` using the method

        :param ary: A `numpy.ndarray` with shape (C, H, W)
        :param method: A `str` of the method to use
        """
        ary = numpy.array(ary)
        func = getattr(self, "_" + method)
        return func(ary)

    def _mean(self, ary):
        """
        Implements a `mean` flattening

        :param ary: A `numpy.ndarray` with shape (C, H, W)

        :returns : A flattened `numpy.ndarray` with shape (H, W)
        """
        return numpy.mean(ary, axis=0)

    def _sum(self, ary):
        """
        Implements a `sum` flattening

        :param ary: A `numpy.ndarray` with shape (C, H, W)

        :returns : A flattened `numpy.ndarray` with shape (H, W)
        """
        return numpy.sum(ary, axis=0)

    def _prod(self, ary):
        """
        Implements a `prod` flattening

        :param ary: A `numpy.ndarray` with shape (C, H, W)

        :returns : A flattened `numpy.ndarray` with shape (H, W)
        """
        return numpy.prod(ary, axis=0)

    def _pca(self, ary):
        """
        Implements a `pca` flattening

        :param ary: A `numpy.ndarray` with shape (C, H, W)

        :returns : A flattened `numpy.ndarray` with shape (H, W)
        """
        if numpy.sum(ary) == 0:
            return numpy.zeros(ary.shape[-2:])
        if len(ary) < 2:
            return ary[0]
        pca = decomposition.PCA(n_components=1, whiten=False, svd_solver="randomized")

        reshaped = numpy.transpose(ary, axes=(1, 2, 0)).reshape(-1, ary.shape[0])

        # transformed = pca.fit_transform(reshaped[:, :-1])
        #
        # avg = numpy.average(transformed.ravel(), weights=reshaped[:, 7])
        #
        # m, M = transformed.min(), transformed.max()
        # transformed -= m
        # transformed /= (M - m)
        #
        # if avg < 0:
        #     transformed -= 1
        #     transformed *= -1

        transformed = pca.fit_transform(reshaped)

        return transformed.reshape(ary.shape[-2:])

class Thresholder:
    def __init__(self):
        pass

    def threshold(self, ary, method, params):
        """
        Thresholds the `numpy.ndarray` using the method

        :param ary : A `numpy.ndarray`
        :param method: A `string` of the thresholding method
        :param params: A
        """
        if isinstance(method, type(None)):
            return ary
        # All zeros
        if not numpy.any(ary):
            return ary
        func = getattr(self, "_" + method)
        return func(ary, params)

    def _otsu(self, ary, params):
        """
        Otsu threshold of an array

        :param ary: A 2D `numpy.ndarray`

        :returns : A thresholded 2D `numpy.ndarray`
        """
        threshold = filters.threshold_otsu(ary)
        return ary >= threshold

    def _triangle(self, ary, params):
        """
        Triangle threshold of an array

        :param ary: A `numpy.ndarray`
        """
        threshold = filters.threshold_triangle(ary)
        return ary >= threshold

    def _percentile(self, ary, params):
        """
        Percentile threshold of an array

        :param ary: A 2D `numpy.ndarray`

        :returns : A thresholded 2D `numpy.ndarray`
        """
        percentile = numpy.percentile(ary, params)
        return ary >= percentile
