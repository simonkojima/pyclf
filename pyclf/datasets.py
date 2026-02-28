import numpy as np

def generate_2d_multiclass_data(
        mus,
        covs,
        n_train,
        n_test,
        random_state=None,
):
    rng = np.random.default_rng(random_state)

    n_classes = len(mus)
    #assert n_classes == 3, "This helper is intended for 3 classes."
    assert len(covs) == n_classes
    assert len(n_train) == n_classes
    assert len(n_test) == n_classes

    Xtr_list, ytr_list = [], []
    Xte_list, yte_list = [], []

    for k in range(n_classes):
        mu = np.asarray(mus[k], dtype=float)
        cov = covs[k]

        if np.isscalar(cov):
            cov_mat = cov * np.eye(2)
        else:
            cov_mat = np.asarray(cov, dtype=float).reshape(2, 2)

        # train
        Xk_tr = rng.multivariate_normal(mean=mu, cov=cov_mat, size=n_train[k])
        yk_tr = np.full(n_train[k], k)

        # test
        Xk_te = rng.multivariate_normal(mean=mu, cov=cov_mat, size=n_test[k])
        yk_te = np.full(n_test[k], k)

        Xtr_list.append(Xk_tr)
        ytr_list.append(yk_tr)
        Xte_list.append(Xk_te)
        yte_list.append(yk_te)

    X_train = np.vstack(Xtr_list)
    y_train = np.concatenate(ytr_list)
    X_test = np.vstack(Xte_list)
    y_test = np.concatenate(yte_list)

    return X_train, y_train, X_test, y_test
