from functools import partial

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    make_scorer,
    mean_squared_error,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import normalize


class BaseSSLScorer:
    def __init__(
        self,
        test_size=0.2,
        random_state=42,
        apply_normalize=True,
        is_multiclass=False,
        **kwargs,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.model_kwargs = kwargs
        self.apply_normalize = apply_normalize
        self.is_multiclass = is_multiclass

    def _get_model(self, **kwargs):
        raise NotImplementedError

    def _get_score_function(self):
        raise NotImplementedError

    def __call__(self, embeddings, target):
        if self.apply_normalize:
            embeddings = normalize(embeddings)
        if self.is_multiclass:
            target = np.argmax(target, axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            embeddings,
            target,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        model = self._get_model(**self.model_kwargs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metric_fn = self._get_score_function()
        score = metric_fn(y_test, y_pred)

        return score


class BaseSSLCVScorer:
    scorer_kwargs = {}

    def __init__(
        self,
        n_splits=5,
        random_state=42,
        apply_normalize=True,
        is_multiclass=False,
        n_jobs=1,
        **kwargs,
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.model_kwargs = kwargs
        self.apply_normalize = apply_normalize
        self.is_multiclass = is_multiclass
        self.n_jobs = n_jobs

    def _get_model(self, **kwargs):
        raise NotImplementedError

    def _get_score_function(self):
        raise NotImplementedError

    def __call__(self, embeddings, target):
        if self.apply_normalize:
            embeddings = normalize(embeddings)
        if self.is_multiclass:
            target = np.argmax(target, axis=1)

        model = self._get_model(**self.model_kwargs)
        metric_fn = self._get_score_function()

        scores = cross_val_score(
            model,
            embeddings,
            target,
            scoring=make_scorer(metric_fn, **self.scorer_kwargs),
            cv=self.n_splits,
            n_jobs=self.n_jobs,
        )
        score = np.mean(scores)

        return score


class AccuracyWithLogisticRegressionCV(BaseSSLCVScorer):
    def _get_model(self, **kwargs):
        return LogisticRegression(**kwargs)

    def _get_score_function(self):
        return accuracy_score


class AccuracyWithLogisticRegression(BaseSSLScorer):
    def _get_model(self, **kwargs):
        return LogisticRegression(**kwargs)

    def _get_score_function(self):
        return accuracy_score


class LogLossWithLogisticRegressionCV(BaseSSLCVScorer):
    def _get_model(self, **kwargs):
        return LogisticRegression(**kwargs)

    def _get_score_function(self):
        return log_loss


class LogLossWithLogisticRegression(BaseSSLScorer):
    def _get_model(self, **kwargs):
        return LogisticRegression(**kwargs)

    def _get_score_function(self):
        return log_loss


class MSEWithLinearRegressionCV(BaseSSLCVScorer):
    def _get_model(self, **kwargs):
        return LinearRegression(**kwargs)

    def _get_score_function(self):
        return mean_squared_error


class MSEWithLinearRegression(BaseSSLScorer):
    def _get_model(self, **kwargs):
        return LinearRegression(**kwargs)

    def _get_score_function(self):
        return mean_squared_error


class RMSEWithLinearRegressionCV(BaseSSLCVScorer):
    scorer_kwargs = {"squared": False}

    def _get_model(self, **kwargs):
        return LinearRegression(**kwargs)

    def _get_score_function(self):
        return mean_squared_error


class RMSEWithLinearRegression(BaseSSLScorer):
    def _get_model(self, **kwargs):
        return LinearRegression(**kwargs)

    def _get_score_function(self):
        return partial(mean_squared_error, squared=False)
