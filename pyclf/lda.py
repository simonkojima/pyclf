import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.covariance import LedoitWolf
from scipy.special import expit
from scipy.linalg import pinv, norm

def subtract_classwise_mean(X, y):
    Xc = X.copy()
    for cl in np.unique(y):
        idx = y == cl
        Xc[idx] -= Xc[idx].mean(axis=0, keepdims=True)
    return Xc


class ShrinkageLDA(BaseEstimator, ClassifierMixin):
    """
    Binary Linear Discriminant Analysis (LDA) with shrinkage covariance estimation.

    This classifier implements a two-class LDA model with a shared covariance matrix
    estimated using shrinkage regularization. The shrinkage parameter can either be
    specified manually or estimated automatically using the Ledoit–Wolf method.

    The model assumes class-conditional Gaussian distributions with class-specific
    means and a common covariance matrix. The decision function is linear in the
    feature space, and class probabilities are obtained via a logistic transform of
    the decision scores.

    Parameters
    ----------
    gamma : float or {'lwf'}, default='lwf'
        Shrinkage coefficient for the pooled covariance matrix.

        - If a float in [0, 1], the covariance is shrunk toward an isotropic
          target matrix according to:
          ``Cw = (1 - gamma) * Sigma + gamma * T``,
          where ``Sigma`` is the pooled within-class covariance and
          ``T = nu * I`` is the scaled identity matrix.
        - If 'lwf', the shrinkage coefficient is estimated automatically using
          the Ledoit–Wolf method.

    priors : {'equal', 'empirical'}, default='equal'
        Class prior probabilities.

        - 'equal': both classes are assigned prior probability 0.5.
        - 'empirical': priors are estimated from the class frequencies in `y`.

    scaling : float or None, default=2
        Optional scaling factor applied to the projection vector `w` so that the
        projected class means are separated by a specified distance. If None,
        no additional scaling is applied.

    inverse : {'inv', 'pinv'}, default='inv'
        Method used to invert the covariance matrix.

        - 'inv': use ``numpy.linalg.inv``.
        - 'pinv': use ``numpy.linalg.pinv`` (more stable for ill-conditioned
          covariance matrices).

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Unique class labels seen during fitting, sorted in ascending order.

    priors_ : list of float
        Prior probabilities for the two classes, in the same order as `classes_`.

    gamma_ : float
        Effective shrinkage coefficient used during fitting. Equals `gamma` if a
        float is provided, or the value estimated by Ledoit–Wolf when
        ``gamma='lwf'``.

    w_ : ndarray of shape (n_features,)
        Linear projection vector defining the discriminant direction.

    b_ : float
        Bias term of the linear decision function.

    Notes
    -----
    The pooled within-class covariance matrix is estimated after subtracting the
    class-wise mean from each sample:

    .. math::

        \\Sigma = \\frac{1}{N - K} \\sum_{c=1}^{K} \\sum_{i \\in c}
        (x_i - \\mu_c)(x_i - \\mu_c)^\\top,

    where :math:`N` is the number of samples and :math:`K=2` is the number of
    classes.

    Shrinkage is then applied toward an isotropic target matrix:

    .. math::

        C_w = (1 - \\gamma)\\,\\Sigma + \\gamma\\,\\nu I,

    with :math:`\\nu = \\mathrm{trace}(\\Sigma) / d` and :math:`d` the number of
    features.

    The decision function is given by:

    .. math::

        f(x) = w^\\top x + b,

    where :math:`w = C_w^{-1}(\\mu_1 - \\mu_0)`.

    Probabilities are obtained by applying the logistic sigmoid to the decision
    scores.

    Raises
    ------
    RuntimeError
        If the number of unique classes in `y` is not equal to 2, or if the
        shrinkage parameter is outside the interval [0, 1].

    References
    ----------
    Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-
    dimensional covariance matrices. Journal of Multivariate Analysis.
    """

    def __init__(
            self,
            gamma="lwf",
            priors="equal",
            scaling=2,
            inverse="inv",
    ):
        self.gamma = gamma
        self.priors = priors
        self.scaling = scaling

        if inverse == "inv":
            self.inverse = np.linalg.inv
        elif inverse == "pinv":
            self.inverse = np.linalg.pinv
        else:
            raise RuntimeError("inverse should be either 'inv' or 'pinv'.")

    def fit(self, X, y=None):

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise RuntimeError("Number of classes should be 2.")

        cl = list()
        for cl_num in self.classes_:
            I = np.where(y == cl_num)[0]
            cl.append(X[I, :])

        mean = list()
        for cl_data in cl:
            mean.append(np.mean(cl_data, axis=0))

        if self.priors == "empirical":
            n0 = cl[0].shape[0]
            n1 = cl[1].shape[0]
            pi0 = n0 / (n0 + n1)
            pi1 = n1 / (n0 + n1)
        elif self.priors == "equal":
            pi0 = pi1 = 0.5
        else:
            raise RuntimeError("priors should be either 'empirical' or 'equal'.")

        self.priors_ = [pi0, pi1]

        Xw = subtract_classwise_mean(X=X, y=y)

        if self.gamma == "lwf":
            lw = LedoitWolf(assume_centered=False).fit(Xw)
            self.gamma_ = lw.shrinkage_
        else:
            self.gamma_ = float(self.gamma)

        if self.gamma_ < 0 or self.gamma_ > 1:
            raise RuntimeError("shrinkage parameter should be in [0, 1].")

        n_samples, n_features = Xw.shape

        Sigma = (Xw.T @ Xw) / (n_samples - 1)

        nu = np.trace(Sigma) / n_features
        T = nu * np.eye(n_features)

        Cw = (1 - self.gamma_) * Sigma + self.gamma_ * T
        Cw_inv = self.inverse(Cw)

        w = Cw_inv @ (mean[1] - mean[0])

        if self.scaling is not None:
            scaling_factor = self.scaling / (w.T @ mean[0] - w.T @ mean[1])
            w = np.squeeze(w * np.absolute(scaling_factor))

        b = -0.5 * (w.T @ mean[0] + w.T @ mean[1]) + np.log(
            self.priors_[1] / self.priors_[0]
        )

        self.w_ = w
        self.b_ = b

        return self

    def decision_function(self, X):
        check_is_fitted(self, ["classes_", "gamma_", "w_", "b_", "priors_"])
        return X @ self.w_ + self.b_

    def predict_proba(self, X):
        d = self.decision_function(X)
        p1 = expit(d)
        p0 = 1 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        d = self.decision_function(X)
        preds = np.where(d >= 0, self.classes_[1], self.classes_[0])

        return preds


class ShrinkageLDA_OVA(BaseEstimator, ClassifierMixin):
    """
    One-vs-All (OvA) extension of Shrinkage Linear Discriminant Analysis (LDA).

    This classifier decomposes a multi-class classification problem into a set of
    binary ShrinkageLDA models, each trained to discriminate one class against
    all remaining classes. During prediction, decision scores from all binary
    models are combined, and the class with the highest score is selected.

    Probabilities are obtained by collecting the positive-class probabilities
    from each binary classifier and normalizing them so that they sum to one
    across classes. These values should be interpreted as heuristic normalized
    OvA scores rather than strictly calibrated multi-class posterior
    probabilities.

    Parameters
    ----------
    gamma : float or {'lwf'}, default='lwf'
        Shrinkage coefficient passed to each underlying :class:`ShrinkageLDA`
        model. If 'lwf', the Ledoit–Wolf method is used to estimate the optimal
        shrinkage parameter for each binary problem.

    priors : {'equal', 'empirical'}, default='equal'
        Class prior strategy used in each binary classifier.

        - 'equal': equal priors for the positive and negative class.
        - 'empirical': priors estimated from class frequencies in the binary
          labels.

    scaling : float or None, default=2
        Optional scaling factor applied to the projection vector in each binary
        ShrinkageLDA model so that projected class means are separated by a
        specified distance. If None, no additional scaling is applied.

    inverse : {'inv', 'pinv'}, default='inv'
        Method used to invert the covariance matrix in each binary model.

        - 'inv': use ``numpy.linalg.inv``.
        - 'pinv': use ``numpy.linalg.pinv`` (more stable for ill-conditioned
          covariance matrices).

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Sorted unique class labels seen during fitting.

    model_dict_ : dict
        Dictionary mapping each class label to its corresponding fitted
        :class:`ShrinkageLDA` binary classifier trained in a one-vs-all fashion.

    Notes
    -----
    For each class :math:`c`, a binary classifier is trained using labels

    .. math::

        y_{bin} = \\mathbb{1}(y = c),

    where the positive class corresponds to samples of class :math:`c` and the
    negative class corresponds to all other samples.

    The decision function returns a matrix of shape ``(n_samples, n_classes)``,
    where each column contains the decision scores of the corresponding binary
    classifier.

    The predicted class is obtained as:

    .. math::

        \\hat{y} = \\arg\\max_c f_c(x),

    where :math:`f_c(x)` is the decision score of the classifier for class
    :math:`c`.

    The probability estimates are computed by extracting the positive-class
    probabilities from each binary classifier and normalizing them:

    .. math::

        \\tilde{p}_c(x) = \\frac{p_c(x)}{\\sum_k p_k(x) + \\epsilon},

    where :math:`p_c(x)` is the OvA probability for class :math:`c` and
    :math:`\\epsilon` is a small constant for numerical stability.

    These normalized values provide relative confidence across classes but are
    not guaranteed to be perfectly calibrated multi-class posterior probabilities.

    """

    def __init__(
            self,
            gamma="lwf",
            priors="equal",
            scaling=2,
            inverse="inv",
    ):

        self.gamma = gamma
        self.priors = priors
        self.scaling = scaling
        self.inverse = inverse

    def fit(self, X, y):

        self.classes_ = np.unique(y)
        self.model_dict_ = {}

        for c in self.classes_:
            y_bin = (y == c).astype(int)

            model = ShrinkageLDA(
                gamma=self.gamma,
                priors=self.priors,
                scaling=self.scaling,
                inverse=self.inverse,
            )

            model.fit(X, y_bin)
            self.model_dict_[c] = model

        return self

    def decision_function(self, X):
        check_is_fitted(self, ["classes_", "model_dict_"])
        output_list = []
        for c in self.classes_:
            output = self.model_dict_[c].decision_function(X)
            output_list.append(output)

        output = np.array(output_list).T

        return output

    def predict(self, X):
        d = self.decision_function(X)
        I = np.argmax(d, axis=1)
        preds = self.classes_[I]
        return preds

    def predict_proba(self, X):
        p_list = []
        for c in self.classes_:
            proba = self.model_dict_[c].predict_proba(X)
            j = np.where(self.model_dict_[c].classes_ == 1)[0][0]
            p_c = proba[:, j]
            p_list.append(p_c)

        p = np.column_stack(p_list)
        p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
        return p


def openvibe_pseudo_inv(X):

    X = 0.5 * (X + X.T)

    eigvals, eigvecs = np.linalg.eigh(X)
    tol = 1e-5 * eigvals[-1]
    mod = eigvals.copy()
    mask = mod >= tol
    mod[mask] = 1.0 / mod[mask]
    X_inv = (eigvecs * mod) @ eigvecs.T

    return X_inv


def openvibe_ledoit_wolf(X, S):

    n_samples, n_features = X.shape

    nu = np.trace(S) / n_features
    F = nu * np.eye(n_features)

    Y = X * X

    phiMat = (Y.T @ Y) / n_samples - S * S

    phi = phiMat.sum()

    gamma = norm(S - F, ord="fro") ** 2
    kappa = phi / gamma

    shrinkage = np.clip(kappa / n_samples, 0.0, 1.0)

    return shrinkage


class OpenVibeLDA(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            shrinkage="openvibe_lwf",
            priors="empirical",
            inverse="openvibe",
            force_diagonal=False,
    ):
        self.shrinkage = shrinkage
        self.priors = priors
        self.inverse = inverse
        self.force_diagonal = force_diagonal

        if inverse == "openvibe":
            self.func_inverse = openvibe_pseudo_inv
        elif inverse == "pinv":
            self.func_inverse = pinv
        else:
            raise RuntimeError("inverse should be either 'openvibe' or 'pinv'")

    def _class_to_index(self, c):
        return self.class_to_index_[c]

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.classes_ = np.unique(y)
        self.mu_ = np.zeros((len(self.classes_), n_features))
        self.w_ = np.zeros((len(self.classes_), n_features))
        self.b_ = np.zeros(len(self.classes_))
        self.Nk_ = np.zeros(len(self.classes_), dtype=int)
        self.priors_ = np.zeros(len(self.classes_))

        self.class_to_index_ = {c: i for i, c in enumerate(self.classes_)}

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

        Xc = X - X.mean(axis=0, keepdims=True)
        S = Xc.T @ Xc / n_samples

        if self.shrinkage == "openvibe_lwf":
            shrinkage = openvibe_ledoit_wolf(Xc, S)
            self.shrinkage_ = shrinkage
        elif self.shrinkage == "lwf":
            lwf = LedoitWolf(assume_centered=True).fit(Xc)
            self.shrinkage_ = lwf.shrinkage_
        else:
            self.shrinkage_ = float(self.shrinkage)

        if self.shrinkage_ < 0 or self.shrinkage_ > 1:
            raise RuntimeError("shrinkage parameter should be in [0, 1].")

        nu = np.trace(S) / n_features
        T = nu * np.eye(n_features)

        S_shrunk = (1 - self.shrinkage_) * S + self.shrinkage_ * T

        if self.force_diagonal:
            S_shrunk = np.diag(np.diag(S_shrunk))

        S_inv = self.func_inverse(S_shrunk)

        for i, c in enumerate(self.classes_):
            mu = self.mu_[i, :]
            self.w_[i, :] = S_inv @ self.mu_[i, :]
            self.b_[i] = (-0.5 * mu.T @ self.w_[i, :]) + np.log(self.priors_[i])

        return self

    def decision_function(self, X):
        check_is_fitted(self)
        return X @ self.w_.T + self.b_[None, :]

    def predict_proba(self, X):
        d = self.decision_function(X)
        n, K = d.shape

        p = np.empty_like(d)
        for i in range(n):
            for k in range(K):
                exp_sum = 0.0
                ak = d[i, k]
                for j in range(K):
                    exp_sum += np.exp(d[i, j] - ak)
                p[i, k] = 1.0 / exp_sum

        return p

    def predict(self, X):
        d = self.decision_function(X)
        I = np.argmax(d, axis=1)
        preds = self.classes_[I]
        return preds


# %%
class OpenVibeLDA_OVA(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            shrinkage="openvibe_lwf",
            priors="empirical",
            inverse="openvibe",
            force_diagonal=False,
    ):

        self.shrinkage = shrinkage
        self.priors = priors
        self.inverse = inverse
        self.force_diagonal = force_diagonal

    def fit(self, X, y):

        self.classes_ = np.unique(y)
        self.model_dict_ = {}

        self.class_to_index_ = {c: i for i, c in enumerate(self.classes_)}

        for i, c in enumerate(self.classes_):
            y_bin = (y != c).astype(int)

            model = OpenVibeLDA(
                shrinkage=self.shrinkage,
                priors=self.priors,
                inverse=self.inverse,
                force_diagonal=self.force_diagonal,
            )

            model.fit(X, y_bin)
            self.model_dict_[i] = model

        return self

    def predict(self, X):
        check_is_fitted(self, ["classes_", "model_dict_"])
        preds_sub = []
        for i, c in enumerate(self.classes_):
            pred = self.model_dict_[i].predict(X)
            preds_sub.append(pred)

        preds_sub = np.stack(preds_sub).T

        preds = []
        for idx, pred_sub in enumerate(preds_sub):

            I = np.where(pred_sub == 0)[0]

            if I.size == 1:
                # there's only one candidate
                preds.append(self.classes_[I[0]])
            elif I.size > 1:
                # there's multiple candidate
                probas = []
                for i in I:
                    proba = self.model_dict_[i].predict_proba(X[idx : idx + 1, :])[0]
                    m = self.model_dict_[i]._class_to_index(0)
                    probas.append(proba[m])
                probas = np.array(probas)
                preds.append(self.classes_[I[np.argmax(probas)]])
            else:
                # no candidate
                probas = []
                for i, c in enumerate(self.classes_):
                    proba = self.model_dict_[i].predict_proba(X[idx : idx + 1, :])[0]
                    m = self.model_dict_[i]._class_to_index(0)
                    probas.append(proba[m])
                probas = np.array(probas)
                preds.append(self.classes_[np.argmax(probas)])

        preds = np.array(preds)
        return preds

    def predict_proba(self, X):
        check_is_fitted(self, ["classes_", "model_dict_"])
        probas = []
        for i, c in enumerate(self.classes_):
            proba = self.model_dict_[i].predict_proba(X)
            probas.append(proba[:, self.model_dict_[i]._class_to_index(0)])
        probas = np.stack(probas).T
        probas /= np.sum(probas, axis=1, keepdims=True)
        return probas

    def decision_function(self, X):
        check_is_fitted(self, ["classes_", "model_dict_"])
        preds = self.predict(X)

        distances = []
        for idx, pred in enumerate(preds):
            sample = X[idx, :]
            d = self.model_dict_[self.class_to_index_[pred]].decision_function(
                sample[None, :]
            )[0]
            distances.append(d)

        distances = np.stack(distances)

        return distances
