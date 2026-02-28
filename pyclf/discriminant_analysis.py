import numpy as np
from scipy.special import expit, softmax, log_softmax
from scipy.linalg import inv, pinv, norm
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.metrics import accuracy_score


def compute_priors(Nk, mode):
    """
    Compute class prior probabilities from class sample counts.

    This utility function converts class-wise sample counts into prior
    probabilities according to the specified strategy. It is intended for use
    in generative classifiers such as LDA, where class priors influence the
    bias term of the discriminant function.

    Parameters
    ----------
    Nk : ndarray of shape (n_classes,)
        Number of samples observed for each class.

    mode : {"empirical", "equal"}
        Strategy used to compute the class priors.
        - "empirical": priors are proportional to the observed class
          frequencies.
        - "equal": all classes are assigned equal prior probability,
          regardless of their frequencies in the data.

    Returns
    -------
    priors : ndarray of shape (n_classes,)
        Array of prior probabilities for each class. The values sum to 1.

    Notes
    -----
    Empirical priors reflect the class distribution in the training data and
    are appropriate when the dataset is representative of the true class
    frequencies. Equal priors can be useful when classes are imbalanced but
    should be treated as equally likely in the model.
    """
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
    """
    Estimate the Ledoit–Wolf shrinkage coefficient for a covariance matrix.

    This function computes the optimal shrinkage coefficient that linearly
    combines the empirical covariance matrix with an isotropic target matrix.
    The implementation follows the classical Ledoit–Wolf approach using a
    data-driven estimate of the shrinkage intensity.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data matrix. Each row corresponds to a sample and each column to
        a feature.

    assume_centered : bool, default=False
        If False, the data are centered by subtracting the column-wise mean
        before computing the covariance statistics. If True, the data are
        assumed to be already centered.

    Returns
    -------
    shrinkage : float
        Estimated shrinkage coefficient in the interval [0, 1]. A value close to
        0 corresponds to little regularization (empirical covariance), while a
        value close to 1 yields a strongly regularized, near-isotropic
        covariance matrix.

    Notes
    -----
    The returned coefficient can be used to form a regularized covariance
    matrix as a convex combination of the empirical covariance and a scaled
    identity matrix. This shrinkage improves numerical stability in
    high-dimensional or small-sample regimes, which are common in BCI and EEG
    applications.
    """

    X = X.copy()

    if assume_centered is False:
        X = X - X.mean(axis=0, keepdims=True)

    n_samples, n_features = X.shape

    S = X.T @ X / n_samples

    nu = np.trace(S) / n_features
    F = nu * np.eye(n_features)

    Y = X * X

    phiMat = (Y.T @ Y) / n_samples - S * S

    phi = phiMat.sum()

    gamma = norm(S - F, ord="fro") ** 2
    if gamma <= 1e-20:
        return 0.0
    kappa = phi / gamma

    shrinkage = np.clip(kappa / n_samples, 0.0, 1.0)

    return shrinkage


def openvibe_pseudo_inv(X):
    """
    Compute a pseudo-inverse using an eigenvalue thresholding strategy inspired by OpenViBE.

    This function computes a symmetric pseudo-inverse of the input matrix by
    performing an eigenvalue decomposition and inverting only the eigenvalues
    above a relative threshold. Small eigenvalues are left unchanged, which
    mimics the behavior used in certain BCI toolchains such as OpenViBE.

    Parameters
    ----------
    X : ndarray of shape (n_features, n_features)
        Symmetric square matrix to be pseudo-inverted. In typical usage, this
        corresponds to a (regularized) covariance matrix.

    Returns
    -------
    X_inv : ndarray of shape (n_features, n_features)
        Pseudo-inverse matrix obtained by selectively inverting eigenvalues
        above a tolerance level.

    Notes
    -----
    The matrix is symmetrized before decomposition to improve numerical
    stability. Eigenvalues below a relative threshold (scaled by the largest
    eigenvalue) are not inverted, which avoids amplification of numerical noise
    in near-singular covariance matrices.

    This behavior differs from the standard Moore–Penrose pseudo-inverse, where
    small eigenvalues are typically set to zero. The present approach is
    intended to reproduce inversion strategies commonly used in practical BCI
    pipelines.
    """

    X = 0.5 * (X + X.T)

    eigvals, eigvecs = np.linalg.eigh(X)
    tol = 1e-5 * eigvals[-1]
    mod = eigvals.copy()
    mask = mod >= tol
    mod[mask] = 1.0 / mod[mask]
    X_inv = (eigvecs * mod) @ eigvecs.T

    return X_inv


class BinaryLinearDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    """
    Binary Linear Discriminant Analysis (LDA) classifier with shared covariance.

    This classifier implements the classical two-class LDA under the Gaussian
    generative model with a shared covariance matrix. The covariance can be
    optionally regularized using Ledoit–Wolf shrinkage and inverted using either
    the matrix inverse or pseudo-inverse.

    The implementation is designed to be simple, readable, and consistent with
    the mathematical formulation of LDA, making it suitable as a reference
    implementation for educational and research purposes.

    Parameters
    ----------
    shrinkage : {"lwf"} or float, default="lwf"
        Shrinkage strategy for the shared covariance matrix.
        - "lwf": Ledoit–Wolf shrinkage is estimated from the data.
        - float: fixed shrinkage coefficient in [0, 1].

    priors : {"empirical", "equal"}, default="empirical"
        Class prior probabilities.
        - "empirical": estimated from class frequencies in the training data.
        - "equal": all classes are assumed to have equal prior probability.

    scaling : float or None, default=2
        Optional scaling factor applied to the weight vector. This rescales the
        magnitude of the discriminant direction without changing its orientation.
        If None, no scaling is applied.

    inverse : {"inv", "pinv"}, default="pinv"
        Method used to invert the (possibly regularized) covariance matrix.
        - "inv": standard matrix inverse (requires full-rank covariance).
        - "pinv": Moore–Penrose pseudo-inverse (robust to rank-deficient cases).

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Sorted unique class labels seen during fitting.

    mu_ : ndarray of shape (2, n_features)
        Estimated class-wise mean vectors.

    priors_ : ndarray of shape (2,)
        Class prior probabilities used in the model.

    shrinkage_ : float
        Estimated or specified shrinkage coefficient applied to the covariance.

    w_ : ndarray of shape (n_features,)
        Weight vector defining the linear discriminant direction.

    b_ : float
        Bias term of the linear discriminant function.

    Notes
    -----
    This implementation assumes exactly two classes. For multi-class problems,
    use :class:`LinearDiscriminantAnalysis`.

    The model follows a generative approach with a shared covariance matrix
    estimated from class-centered samples. Shrinkage regularization can improve
    stability in small-sample or high-dimensional settings, which are common in
    BCI and EEG applications.
    """

    def __init__(
        self,
        shrinkage="lwf",
        priors="empirical",
        scaling=2,
        inverse="pinv",
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
            self.shrinkage_ = shrinkage
        else:
            self.shrinkage_ = float(self.shrinkage)

        if self.shrinkage_ < 0 or self.shrinkage_ > 1:
            raise RuntimeError("shrinkage parameter should be in [0, 1].")

        nu = np.trace(Sw) / n_features
        T = nu * np.eye(n_features)

        Sw_shrunk = (1 - self.shrinkage_) * Sw + self.shrinkage_ * T

        if self.inverse == "inv":
            self._inv = inv
        elif self.inverse == "pinv":
            self._inv = pinv
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
        check_is_fitted(self, ["classes_", "shrinkage_", "w_", "b_"])
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
    """
    Multi-class Linear Discriminant Analysis (LDA) classifier with shared covariance.

    This classifier implements the classical multi-class LDA under a Gaussian
    generative model where all classes share a common covariance matrix. Each
    class is modeled by a linear discriminant function, and class probabilities
    are obtained via a softmax transformation of these discriminant scores.

    The implementation emphasizes clarity and direct correspondence with the
    standard LDA formulation, making it suitable as a reference implementation
    for educational and research use.

    Parameters
    ----------
    shrinkage : {"lwf"} or float, default="lwf"
       Shrinkage strategy for the shared covariance matrix.
       - "lwf": Ledoit–Wolf shrinkage estimated from the data.
       - float: fixed shrinkage coefficient in [0, 1].

    priors : {"empirical", "equal"}, default="empirical"
       Class prior probabilities.
       - "empirical": estimated from class frequencies in the training data.
       - "equal": all classes are assumed to have equal prior probability.

    inverse : {"inv", "pinv", "openvibe"}, default="pinv"
       Method used to invert the shared covariance matrix.
       - "inv": standard matrix inverse (requires full-rank covariance).
       - "pinv": Moore–Penrose pseudo-inverse (robust to rank-deficient cases).
       - "openvibe": pseudo-inverse with eigenvalue thresholding similar to
         the OpenViBE implementation.

    covariance : {"within", "global"}, default="within"
       Strategy used to estimate the shared covariance matrix.
       - "within": class-wise centering (within-class covariance).
       - "global": centering using the global mean of all samples.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
       Sorted unique class labels seen during fitting.

    mu_ : ndarray of shape (n_classes, n_features)
       Estimated class-wise mean vectors.

    priors_ : ndarray of shape (n_classes,)
       Class prior probabilities used in the model.

    shrinkage_ : float
       Estimated or specified shrinkage coefficient applied to the covariance.

    w_ : ndarray of shape (n_classes, n_features)
       Weight vectors defining the linear discriminant functions for each class.

    b_ : ndarray of shape (n_classes,)
       Bias terms of the class-specific discriminant functions.

    Notes
    -----
    This implementation follows the standard generative multi-class LDA with a
    shared covariance matrix. The discriminant scores are transformed into class
    probabilities using a softmax function, ensuring numerically stable
    probability estimates.

    The option ``covariance="within"`` corresponds to the classical pooled
    within-class covariance used in LDA, while ``covariance="global"`` uses a
    globally centered covariance matrix for alternative modeling assumptions.
    """

    def __init__(
        self,
        shrinkage="lwf",
        priors="empirical",
        inverse="pinv",
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
            self._inv = inv
        elif self.inverse == "pinv":
            self._inv = pinv
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
        check_is_fitted(self, ["classes_", "shrinkage_", "w_", "b_"])
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
