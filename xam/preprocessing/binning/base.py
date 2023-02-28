import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array


class BaseBinner(BaseEstimator, TransformerMixin):

    @property
    def cut_points(self):
        raise NotImplementedError

    def transform(self, X, y=None):
        """Binarize X based on the fitted cut points."""

        # scikit-learn checks
        X = check_array(X)

        if self.cut_points is None:
            raise NotFittedError('Estimator not fitted, call `fit` before exploiting the model.')

        if X.shape[1] != len(self.cut_points):
            raise ValueError("Provided array's dimensions do not match with the ones from the "
                             "array `fit` was called on.")

        binned = np.array([
            np.digitize(x, self.cut_points[i])
            if len(self.cut_points[i]) > 0
            else np.zeros(x.shape)
            for i, x in enumerate(X.T)
        ]).T

        return binned


class BaseSupervisedBinner(BaseBinner):

    def fit(X, y, **fit_params):
        raise NotImplementedError


class BaseUnsupervisedBinner(BaseBinner):

    def fit(X, y=None, **fit_params):
        raise NotImplementedError
