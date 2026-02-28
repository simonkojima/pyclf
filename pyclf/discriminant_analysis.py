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


class ShrinkageLDA_OVA(BaseEstimator, ClassifierMixin):
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
