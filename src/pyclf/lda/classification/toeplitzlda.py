from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.utils.multiclass
from blockmatrix import (
    SpatioTemporalMatrix,
    fortran_block_levinson,
    fortran_cov_mean_transformation,
    linear_taper,
)
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted, check_X_y

from .covariance import (
    calc_n_times,
    shrinkage,
    subtract_classwise_means,
)

from mne import BaseEpochs, concatenate_epochs, epochs

class EpochsVectorizer(BaseEstimator, TransformerMixin):

    """
        Original code of this class by implemented by Jan Sosulski. Modified by Simon Kojima.

        --
        https://github.com/jsosulski/toeplitzlda
        Copyright (c) 2022 Jan Sosulski
        All rights reserved.

        Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the limitations in the disclaimer below) provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
        * Neither the name of Jan Sosulski nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

        NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """

    def __init__(
        self,
        sfreq = None,
        t_ref = None,
        permute_channels_and_time=True,
        select_ival=None,
        jumping_mean_ivals=None,
        averaging_samples=None,
        rescale_to_uv=False,
        mne_scaler=None,
        pool_times=False,
        to_np_only=False,
        copy=True
    ):

        self.sfreq = sfreq
        self.t_ref = t_ref
        self.permute_channels_and_time = permute_channels_and_time
        self.jumping_mean_ivals = jumping_mean_ivals
        self.select_ival = select_ival
        self.averaging_samples = averaging_samples
        self.rescale_to_uv = rescale_to_uv
        self.scaling = 1e6 if self.rescale_to_uv else 1
        self.pool_times = pool_times
        self.to_np_only = to_np_only
        self.copy = copy
        self.mne_scaler = mne_scaler
        if self.select_ival is None and self.jumping_mean_ivals is None:
            raise ValueError("jumping_mean_ivals or select_ival is required")

    def fit(self, X, y=None):
        """fit."""
        return self

    def transform(self, X):
        """transform."""
        e = X.copy() if self.copy else X
        type_e = type(e)
        if type_e is epochs.EpochsFIF:
            type_e = BaseEpochs
        elif type_e is epochs.Epochs:
            type_e = BaseEpochs
        elif type_e is epochs.EpochsArray:
            type_e = BaseEpochs

        if type_e is not BaseEpochs and type_e is not np.ndarray:
            if type_e is list and type(e[0]) is BaseEpochs:
                e = concatenate_epochs(e, add_offset=False) # Is it ok to fix this to add_offset=Fasle ?
            else:
                raise ValueError("argument X has unknown type : " + str(type_e))
        if self.to_np_only:
            if type_e is BaseEpochs:
                X = e.get_data() * self.scaling
                return X
            else:
                raise ValueError("argument X is already np_array")
        if self.jumping_mean_ivals is not None:
            self.averaging_samples = np.zeros(len(self.jumping_mean_ivals))
            if type_e is BaseEpochs:
                X = e.get_data() * self.scaling
            else:
                X = e * self.scaling
                if self.sfreq is None:
                    raise ValueError("specify the sampling frequency")
                if self.t_ref is None:
                    raise ValueError("specify the time reference")
            new_X = np.zeros((X.shape[0], X.shape[1], len(self.jumping_mean_ivals)))
            for i, ival in enumerate(self.jumping_mean_ivals):
                if type_e is BaseEpochs:
                    np_idx = e.time_as_index(ival)
                else:
                    np_idx = time_as_index(ival, self.t_ref, self.sfreq)
                idx = list(range(np_idx[0], np_idx[1]))
                self.averaging_samples[i] = len(idx)
                new_X[:, :, i] = np.mean(X[:, :, idx], axis=2)
            X = new_X
        elif self.select_ival is not None:
            if type_e is BaseEpochs:
                e.crop(tmin=self.select_ival[0], tmax=self.select_ival[1])
                X = e.get_data() * self.scaling
            else:
                if self.sfreq is None:
                    raise ValueError("specify the sampling frequency")
                if self.t_ref is None:
                    raise ValueError("specify the time reference")
                X = e * self.scaling
                t_idx = time_as_index(self.select_ival, self.t_ref, self.sfreq)
                X = X[:,:,t_idx[0]:(t_idx[1]+1)] # t_idx[1]+1 to be same as e.crop()
        elif self.pool_times:
            if type_e is BaseEpochs:
                X = e.get_data() * self.scaling
            else:
                X = e * self.scaling
            raise ValueError("This should never be entered though.")
        else:
            raise ValueError(
                "In the constructor, pass either select ival or jumping means."
            )
        if self.mne_scaler is not None:
            X = self.mne_scaler.fit_transform(X)
        if self.permute_channels_and_time and not self.pool_times:
            X = X.transpose((0, 2, 1))
        if self.pool_times:
            X = np.reshape(X, (-1, X.shape[1]))
        else:
            X = np.reshape(X, (X.shape[0], -1))
        return X

def time_as_index(times, t_ref, sfreq, use_rounding=False):

    """Convert time to indices.
    Parameters
    ----------
    times : list-like | float | int
        List of numbers or a number representing points in time.
    t_ref : float | int
        time reference to compute index of each time point.
        usually, it is the first time index of concerned data.
    sfreq : float | int
        sampling frequency of concerned data.
    use_rounding : bool
        If True, use rounding (instead of truncation) when converting
        times to indices. This can help avoid non-unique indices.
    Returns
    -------
    index : ndarray
        Indices corresponding to the times supplied.

    ---
    This function is originally from MNE-Python project.
    Modified by Simon Kojima

    Copyright ?? 2011-2022, authors of MNE-Python
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the copyright holder nor the names of its
        contributors may be used to endorse or promote products derived from
        this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """

    index = (np.atleast_1d(times) - t_ref) * sfreq
    if use_rounding:
        index = np.round(index)
    return index.astype(int)

class ShrinkageLinearDiscriminantAnalysis(
    ClassifierMixin,
    BaseEstimator,
):
    """SLDA Implementation with bells and whistles and a lot of options."""

    def __init__(
        self,
        priors=None,
        only_block=False,
        n_times="infer",
        n_channels=31,
        pool_cov=True,
        standardize_shrink=True,
        calculate_oracle_mean=None,
        unit_w=False,
        fixed_gamma=None,
        enforce_toeplitz=False,
        use_fortran_solver=False,
        banding=None,
        tapering=None,
        data_is_channel_prime=True,
    ):
        self.only_block = only_block
        self.priors = priors
        self.n_times = n_times
        self.n_channels = n_channels
        self.pool_cov = pool_cov
        self.standardize_shrink = standardize_shrink
        self.calculate_oracle_mean = calculate_oracle_mean
        self.unit_w = unit_w
        self.fixed_gamma = fixed_gamma
        self.enforce_toeplitz = enforce_toeplitz
        self.use_fortran_solver = use_fortran_solver
        if self.use_fortran_solver and not self.enforce_toeplitz:
            raise ValueError("Can only use Fortran solver when enforce_toeplitz=True")
        self.banding = banding
        self.tapering = tapering
        self.data_is_channel_prime = data_is_channel_prime

        # added for adoptation
        self.C_inv = None
        self.cl_mean = None
        self.y_train = None
        self.prior_offset = None

    def fit(self, X_train, y, oracle_data=None):
        # Section: Basic setup
        check_X_y(X_train, y)
        self.classes_ = sklearn.utils.multiclass.unique_labels(y)
        if self.calculate_oracle_mean is None:
            oracle_data = None
        xTr = X_train.T

        self.y_train = y

        if self.priors is None:
            # here we deviate from the bbci implementation and
            # use the sample priors by default
            _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
            priors = np.bincount(y_t) / float(len(y))
        else:
            priors = self.priors

        # Section: covariance / mean calculation
        X, cl_mean = subtract_classwise_means(xTr, y)
        # Check if mean information is provided and if so whether to use it only for class-wise
        # mean estimation or also for the subtraction of class means from the data to estimate the
        # covariance matrix
        if self.calculate_oracle_mean == "clmean_and_covmean":
            _, cl_mean = subtract_classwise_means(oracle_data["x"].T, oracle_data["y"])
            X, _ = subtract_classwise_means(xTr, y, ext_mean=cl_mean)
        elif self.calculate_oracle_mean == "only_clmean":
            _, cl_mean = subtract_classwise_means(oracle_data["x"].T, oracle_data["y"])
        # Subtract class-wise means from data and then estimate the covariance or average
        # class-wise covariance matrices?
        if self.pool_cov:
            C_cov, C_gamma = shrinkage(
                X,
                standardize=self.standardize_shrink,
                gamma=self.fixed_gamma,
            )
        else:
            C_cov = np.zeros((xTr.shape[0], xTr.shape[0]))
            for cur_class in self.classes_:
                class_idxs = y == cur_class
                x_slice = X[:, class_idxs]
                C_cov += priors[cur_class] * shrinkage(x_slice)[0]
        dim = C_cov.shape[0]
        nt = calc_n_times(dim, self.n_channels, self.n_times)
        stm = SpatioTemporalMatrix(
            C_cov,
            n_chans=self.n_channels,
            n_times=nt,
            channel_prime=self.data_is_channel_prime,
        )
        if not self.data_is_channel_prime:
            stm.swap_primeness()
        if self.enforce_toeplitz:
            stm.force_toeplitz_offdiagonals()
        # Banding could be realized with a binary taper, so remove?
        if self.banding is not None:
            stm.band_offdiagonals(self.banding)
        if self.tapering is not None:
            stm.taper_offdiagonals(taper_f=self.tapering)
        if not self.data_is_channel_prime:
            stm.swap_primeness()

        self.stored_cl_mean = cl_mean
        self.stored_stm = stm
        C_cov = stm.mat

        if self.only_block:
            C_cov_new = np.zeros_like(C_cov)
            for i in range(nt):
                idx_start = i * self.n_channels
                idx_end = idx_start + self.n_channels
                C_cov_new[idx_start:idx_end, idx_start:idx_end] = C_cov[
                    idx_start:idx_end, idx_start:idx_end
                ]
            C_cov = C_cov_new

        prior_offset = np.log(priors)

        # Numpy uses more cores, fortran library only one
        # with threadpoolctl.threadpool_limits(limits=1):
        if self.use_fortran_solver:
            # TODO implement transformation for mean.ndim == 2 in block_matrix.py
            w = np.empty_like(cl_mean)
            self.fit_time_ = 0
            for ci in range(w.shape[1]):
                fcov, fmean = fortran_cov_mean_transformation(
                    C_cov, cl_mean[:, ci], nch=self.n_channels, ntim=nt
                )
                st = time.time()
                w[:, ci] = fortran_block_levinson(fcov, fmean, transform_A=False)
                self.fit_time_ += time.time() - st
        else:
            st = time.time()
            w = np.linalg.solve(C_cov, cl_mean)
            self.fit_time_ = time.time() - st
        
        w = w / np.linalg.norm(w) if self.unit_w else w
        b = -0.5 * np.sum(cl_mean * w, axis=0).T + prior_offset

        self.coef_ = w.T
        self.intercept_ = b
        
        self.C_inv = np.linalg.inv(C_cov)
        self.cl_mean = cl_mean

        return C_cov, cl_mean, prior_offset
    
    def adaptation(self, X, y, eta_cov = 0.001, eta_mean = 0.005):
        """
        Calculate new w and b by adaptation method proposed by Vidaurre et al.(2011)[1]

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The feature matrix to be feeded for adaptation.


        [1] Vidaurre et al. Toward Unsupervised Adaptation of LDA for Brain_Computer Interfaces. (2011)
        
        """
        
        y_train = np.append(self.y_train, y)
        C_inv = self.C_inv
        cl_mean = self.cl_mean

        for idx, cl in enumerate(y):
            
            x = X[idx, :]
            x = x.reshape((x.size, 1))

            # Covariance adaptation
            v = np.dot(C_inv, x)
            
            frac_num = np.dot(v, v.T)
            frac_den = ((1 - eta_cov)/eta_cov) + np.dot(x.T, v)
            
            I = C_inv - (frac_num / frac_den)
            C_inv = I / (1 - eta_cov)

            # Class mean adaptation
            class_idx = np.where(self.classes_ == cl)[0]
            
            shape = cl_mean[:, class_idx].shape

            cl_mean[:,class_idx] = (1-eta_mean)*cl_mean[:,class_idx] + (eta_mean*x)
         
            
        if self.priors is None:
            # here we deviate from the bbci implementation and
            # use the sample priors by default
            _, y_t = np.unique(y_train, return_inverse=True)  # non-negative ints
            priors = np.bincount(y_t) / float(len(y))
        else:
            priors = self.priors
        prior_offset = np.log(priors)

        self.C_inv = C_inv
        self.cl_mean = cl_mean
        self.y_train = y_train
        self.prior_offset = prior_offset
    
    def apply_adaptation(self):

        w = np.dot(self.C_inv, self.cl_mean)
        w = w / np.linalg.norm(w) if self.unit_w else w
        b = -0.5 * np.sum(self.cl_mean * w, axis=0).T + self.prior_offset

        self.coef_ = w.T
        self.intercept_ = b

    def set_parameters(self, coef, intercept, C_inv, cl_mean, y_train, classes):
        self.coef_ = coef
        self.intercept_ = intercept
        self.C_inv = C_inv
        self.cl_mean = cl_mean
        self.y_train = y_train
        self.classes_ = classes

    def decision_function(self, X):
        """
        Predict confidence scores for samples.

        The confidence score for a sample is proportional to the signed
        distance of that sample to the hyperplane.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data matrix for which we want to get the confidence scores.

        Returns
        -------
        scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Confidence scores per `(n_samples, n_classes)` combination. In the
            binary case, confidence score for `self.classes_[1]` where >0 means
            this class would be predicted.
        """
        check_is_fitted(self)

        X = self._validate_data(X, accept_sparse="csr", reset=False)
        scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
        if scores.shape[1] == 2:
            scores = scores[:, 1] - scores[:, 0]
        return scores.squeeze()

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data matrix for which we want to get the predictions.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the class labels for each sample.
        """
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def predict_proba(self, X):
        """Estimate probability.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            Estimated probabilities.
        """
        prob = self.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)

        return np.column_stack([1 - prob, prob])

    def predict_log_proba(self, X):
        """Estimate log probability.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            Estimated log probabilities.
        """
        return np.log(self.predict_proba(X))


class ToeplitzLDA(ShrinkageLinearDiscriminantAnalysis):
    def __init__(
        self,
        n_channels=None,
        unit_w=False,
        *,
        n_times="infer",
        data_is_channel_prime=True,
        use_fortran_solver=False,
    ):
        if n_channels is None:
            raise ValueError(f"Required parameter n_channels is not set.")
        super().__init__(
            data_is_channel_prime=data_is_channel_prime,
            n_times=n_times,
            n_channels=n_channels,
            use_fortran_solver=use_fortran_solver,
            tapering=linear_taper,
            enforce_toeplitz=True,
            unit_w=unit_w,
        )


# This is a very simple implementation of sLDA for didactic purposes
class PlainLDA(BaseEstimator, ClassifierMixin):
    """Straightforward SLDA implementation mostly for ease of understanding.

    In contrast to the other LDAs in this file, this implementation actually uses the bias for
    class prediction. The other classifiers report the class with maximal classifier output, i.e.,
    output from decision_function."""

    def __init__(
        self,
        toeplitz_time=False,
        taper_time=None,
        use_fortran_solver=False,
        n_times=None,
        n_channels=None,
        global_cov=False,
    ):
        self.w = None
        self.b = None
        self.n_times = n_times
        self.n_channels = n_channels
        self.toeplitz_time = toeplitz_time
        self.taper_time = taper_time
        self.use_fortran_solver = use_fortran_solver
        self.global_cov = global_cov

        self.mu_T = None
        self.mu_NT = None

        self.stm_info = None

    def fit(self, X, y):
        """
        Parameters
        ----------
        X: np.ndarray
            Input data of shape (n_samples, n_chs, n_time)
        y: np.ndarray
            Actual labels of X
        """

        X = X.reshape(X.shape[0], -1)
        self.classes_ = np.unique(y)
        if set(self.classes_) != {0, 1}:
            raise ValueError(
                "This LDA only supports binary classification. Use one of the other classifiers "
                "for multi-class settings."
            )

        mu_T = np.mean(X[np.where(y == 1)], axis=0)
        mu_NT = np.mean(X[np.where(y == 0)], axis=0)

        # Subtract class-means unless we want to ignore them
        if not self.global_cov:
            X = subtract_classwise_means(X.T, y)[0].T
        C_cov, gamma = shrinkage(X.T)

        # ToeplitzLDA specific code BEGIN
        nt = calc_n_times(C_cov.shape[0], self.n_channels, self.n_times)
        stm = SpatioTemporalMatrix(C_cov, n_chans=nt, n_times=self.n_times)

        if self.toeplitz_time:
            stm.force_toeplitz_offdiagonals()
        if self.taper_time is not None:
            stm.taper_offdiagonals(self.taper_time)

        self.stored_stm = stm
        C_cov = stm.mat
        # ToeplitzLDA specific code END

        C_diff = mu_T - mu_NT
        C_mean = 0.5 * (mu_T + mu_NT)

        if self.use_fortran_solver:
            if not self.toeplitz_time:
                raise ValueError("Cannot use fortran solver without block-Toeplitz structure")
            C_w = fortran_block_levinson(C_cov, C_diff, nch=self.n_channels, ntim=self.n_times)
        else:
            C_w = np.linalg.solve(C_cov, C_diff)
        C_b = np.dot(-C_w.T, C_mean)

        self.w = C_w.reshape((1, -1))
        self.b = C_b

        self.mu_T = mu_T
        self.mu_NT = mu_NT

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = X.reshape(X.shape[0], -1)
        return np.dot(X, self.w.T) + self.b

    def predict(self, X: np.ndarray):
        return self.decision_function(X) > 0
