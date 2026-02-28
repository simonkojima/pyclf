import numpy as np


def generate_2d_data(
    mus,
    covs,
    n_train,
    n_test,
    random_state=None,
):
    rng = np.random.default_rng(random_state)

    n_classes = len(mus)
    assert len(covs) == n_classes
    assert len(n_train) == n_classes
    assert len(n_test) == n_classes

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    for k in range(n_classes):
        mu = np.asarray(mus[k], dtype=float)
        cov = covs[k]

        if np.isscalar(cov):
            cov_mat = cov * np.eye(2)
        else:
            cov_mat = np.asarray(cov, dtype=float).reshape(2, 2)

        # train
        Xk_train = rng.multivariate_normal(mean=mu, cov=cov_mat, size=n_train[k])
        yk_train = np.full(n_train[k], k)

        # test
        Xk_test = rng.multivariate_normal(mean=mu, cov=cov_mat, size=n_test[k])
        yk_test = np.full(n_test[k], k)

        X_train_list.append(Xk_train)
        y_train_list.append(yk_train)
        X_test_list.append(Xk_test)
        y_test_list.append(yk_test)

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)

    return X_train, y_train, X_test, y_test
