"""
Example 01: LDA on artificial 3-class data
==========================================
"""

# Authors: Simon Kojima <simon.kojima@inria.fr>
#
# License: BSD (3-clause)

# %%
from pyclf.discriminant_analysis import LinearDiscriminantAnalysis
from pyclf.datasets import generate_2d_data
from pyclf.viz import plot_2d_scatter

from sklearn.metrics import accuracy_score

# %%

mus = [
    [0.0, 1.0],
    [-1.0, -1.0],
    [1.0, -1.0],
]

covs = [
    0.5,
    0.5,
    0.5,
]

n_train = [500, 500, 500]
n_test = [100, 100, 100]

X_train, y_train, X_test, y_test = generate_2d_data(
    mus=mus,
    covs=covs,
    n_train=n_train,
    n_test=n_test,
    random_state=42,
)


# %%

plot_2d_scatter(X_train, y_train, X_test, y_test)

# %%

model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)

preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)
print(f"Accuracy: {acc:.2f}")
