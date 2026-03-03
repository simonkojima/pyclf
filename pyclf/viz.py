import numpy as np
import matplotlib.pyplot as plt


def plot_2d_scatter(X_train, y_train, X_test, y_test, title="2D Multiclass Data"):
    classes = np.unique(y_test)
    colors = ["tab:blue", "tab:orange", "tab:green"]

    plt.figure(figsize=(6, 6))

    for k in classes:
        # train
        idx_tr = y_test == k
        plt.scatter(
            X_train[idx_tr, 0],
            X_train[idx_tr, 1],
            c=colors[k],
            marker="o",
            alpha=0.6,
            label=f"train class {k}" if k == classes[0] else None,
        )

        # test
        idx_te = y_test == k
        plt.scatter(
            X_test[idx_te, 0], X_test[idx_te, 1], c=colors[k], marker="x", alpha=0.9
        )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()
