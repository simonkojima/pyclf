import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.covariance import LedoitWolf
from scipy.special import expit, softmax, log_softmax
from scipy.linalg import inv, pinv, norm


def compute_priors(Nk, mode):
    n_classes = Nk.size
    n_samples = Nk.sum()

    if mode == "empirical":
        priors = Nk / n_samples
    elif mode == "equal":
        priors = np.full(n_classes, 1.0 / n_classes)
    else:
        raise RuntimeError("priors should be either 'empirical' or 'equal'.")

    return priors


def ledoit_wolf(X, assume_centered=False):

    X = X.copy()

    if assume_centered is False:
        X = X - X.mean(axis=0)

    n_samples, n_features = X.shape

    S = X.T @ X / n_samples

    nu = np.trace(S) / n_features
    F = nu * np.eye(n_features)

    Y = X * X

    phiMat = (Y.T @ Y) / n_samples - S * S

    phi = phiMat.sum()

    gamma = norm(S - F, ord="fro") ** 2
    kappa = phi / gamma

    shrinkage = np.clip(kappa / n_samples, 0.0, 1.0)

    return shrinkage


def openvibe_pseudo_inv(X):

    X = 0.5 * (X + X.T)

    eigvals, eigvecs = np.linalg.eigh(X)
    tol = 1e-5 * eigvals[-1]
    mod = eigvals.copy()
    mask = mod >= tol
    mod[mask] = 1.0 / mod[mask]
    X_inv = (eigvecs * mod) @ eigvecs.T

    return X_inv


class BinaryLinearDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        shrinkage="lwf",
        priors="empirical",
        scaling=2,
        inverse="inv",
    ):
        self.shrinkage = shrinkage
        self.priors = priors
        self.scaling = scaling
        self.inverse = inverse

    def fit(self, X, y=None):

        X, y = check_X_y(X, y)

        n_samples, n_features = X.shape

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise RuntimeError("Number of classes should be 2.")

        self.mu_ = np.zeros((2, n_features))
        self.w_ = np.zeros(n_features)
        self.b_ = None
        self.Nk_ = np.zeros(2, dtype=int)
        self.priors_ = np.zeros(2)

        for idx_c, c in enumerate(self.classes_):
            idx = y == c
            self.Nk_[idx_c] = int(idx.sum())
            self.mu_[idx_c, :] = X[idx].mean(axis=0)

        self.priors_ = compute_priors(self.Nk_, self.priors)

        Xc = X.copy()

        # compute class-wise covariace
        for idx_c, c in enumerate(self.classes_):
            idx = y == c
            Xc[idx, :] = Xc[idx, :] - self.mu_[idx_c, :]

        Sw = (Xc.T @ Xc) / n_samples

        if self.shrinkage == "lwf":
            shrinkage = ledoit_wolf(
                X=Xc,
                assume_centered=False,
            )
            self.gamma_ = shrinkage
        else:
            self.gamma_ = float(self.gamma)

        if self.gamma_ < 0 or self.gamma_ > 1:
            raise RuntimeError("shrinkage parameter should be in [0, 1].")

        nu = np.trace(Sw) / n_features
        T = nu * np.eye(n_features)

        Sw_shrunk = (1 - self.gamma_) * Sw + self.gamma_ * T

        if self.inverse == "inv":
            self._inv = np.linalg.inv
        elif self.inverse == "pinv":
            self._inv = np.linalg.pinv
        else:
            raise RuntimeError("inverse should be either 'inv' or 'pinv'.")

        Sw_shrunk_inv = self._inv(Sw_shrunk)

        w = Sw_shrunk_inv @ (self.mu_[1] - self.mu_[0])

        if self.scaling is not None:
            scaling_factor = self.scaling / (w.T @ self.mu_[0] - w.T @ self.mu_[1])
            w = np.squeeze(w * np.absolute(scaling_factor))

        b = -0.5 * (w.T @ self.mu_[0] + w.T @ self.mu_[1]) + np.log(
            self.priors_[1] / self.priors_[0]
        )

        self.w_ = w
        self.b_ = b

        return self

    def decision_function(self, X):
        check_is_fitted(self, ["classes_", "gamma_", "w_", "b_"])
        X = check_array(X)
        return X @ self.w_ + self.b_

    def predict_proba(self, X):
        d = self.decision_function(X)
        p1 = expit(d)
        p0 = 1 - p1
        return np.column_stack([p0, p1])

    def predict_log_proba(self, X):
        proba = self.predict_proba(X)
        return np.log(proba)

    def predict(self, X):
        d = self.decision_function(X)
        preds = np.where(d >= 0, self.classes_[1], self.classes_[0])
        return preds

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


class LinearDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        shrinkage="lwf",
        priors="empirical",
        inverse="inv",
        covariance="within",
    ):
        self.shrinkage = shrinkage
        self.priors = priors
        self.inverse = inverse
        self.covariance = covariance

    def fit(self, X, y):

        X, y = check_X_y(X, y)

        n_samples, n_features = X.shape

        self.classes_ = np.unique(y)
        self.mu_ = np.zeros((len(self.classes_), n_features))
        self.w_ = np.zeros((len(self.classes_), n_features))
        self.b_ = np.zeros(len(self.classes_))
        self.Nk_ = np.zeros(len(self.classes_), dtype=int)
        self.priors_ = np.zeros(len(self.classes_))

        for i, c in enumerate(self.classes_):
            idx = y == c
            self.Nk_[i] = int(idx.sum())
            self.mu_[i, :] = X[idx].mean(axis=0)

        # priors
        if self.priors == "empirical":
            self.priors_ = self.Nk_ / n_samples
        elif self.priors == "equal":
            self.priors_ = np.full(len(self.classes_), 1.0 / len(self.classes_))
        else:
            raise RuntimeError("priors should be either 'empirical' or 'equal'.")

        if self.covariance == "within":
            Xc = np.zeros_like(X)
            for idx_c, c in enumerate(self.classes_):
                idx = y == c
                Xc[idx, :] = X[idx, :] - self.mu_[idx_c, :]
        elif self.covariance == "global":
            Xc = X - X.mean(axis=0, keepdims=True)
        else:
            raise RuntimeError("covariance should be either 'within' or 'global'.")

        S = Xc.T @ Xc / n_samples

        if self.shrinkage == "lwf":
            shrinkage = ledoit_wolf(Xc, assume_centered=False)
            self.shrinkage_ = shrinkage
        else:
            self.shrinkage_ = float(self.shrinkage)

        if self.shrinkage_ < 0 or self.shrinkage_ > 1:
            raise RuntimeError("shrinkage parameter should be in [0, 1].")

        nu = np.trace(S) / n_features
        T = nu * np.eye(n_features)

        S_shrunk = (1 - self.shrinkage_) * S + self.shrinkage_ * T

        if self.inverse == "inv":
            self._inv = np.linalg.inv
        elif self.inverse == "pinv":
            self._inv = np.linalg.pinv
        elif self.inverse == "openvibe":
            self._inv = openvibe_pseudo_inv
        else:
            raise RuntimeError("inverse should be either 'inv', 'pinv', or 'openvibe'")

        S_inv = self._inv(S_shrunk)

        for i, c in enumerate(self.classes_):
            mu = self.mu_[i, :]
            self.w_[i, :] = S_inv @ self.mu_[i, :]
            self.b_[i] = (-0.5 * mu.T @ self.w_[i, :]) + np.log(self.priors_[i])

        return self

    def decision_function(self, X):
        X = check_array(X)
        check_is_fitted(self)
        return X @ self.w_.T + self.b_[None, :]

    def predict_proba(self, X):
        d = self.decision_function(X)

        p = softmax(d, axis=1)

        return p

    def predict_log_proba(self, X):
        d = self.decision_function(X)
        p = log_softmax(d, axis=1)
        return p

    def predict(self, X):
        d = self.decision_function(X)
        I = np.argmax(d, axis=1)
        preds = self.classes_[I]
        return preds

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
