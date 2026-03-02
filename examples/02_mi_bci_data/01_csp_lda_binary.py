"""
Example 01: CSP + LDA on binary motor imagery data
==================================================
"""

# Authors: Simon Kojima <simon.kojima@inria.fr>
#
# License: BSD (3-clause)

# %%
from pyclf.discriminant_analysis import BinaryLinearDiscriminantAnalysis
from pyclf.utils import labels_from_epochs

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import mne
from moabb.datasets import Dreyer2023

# %%
dataset = Dreyer2023()

subject = 10
l_freq, h_freq = 8, 30
f_order = 4
tmin, tmax = dataset.interval[0] + 0.5, dataset.interval[1] - 0.5
resample = 128

data = dataset.get_data(subjects=[subject])[subject]["0"]

epochs_train, epochs_test = [], []
for name, raw in data.items():
    raw.pick(picks="eeg")

    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="iir",
        iir_params={"ftype": "butter", "btype": "bandpass", "order": f_order},
        phase="zero",
    )

    epochs = mne.Epochs(
        raw, tmin=tmin, tmax=tmax, baseline=None, preload=True, event_repeated="merge"
    )

    epochs = epochs[["left_hand", "right_hand"]]

    if "acquisition" in name:
        epochs_train.append(epochs)
    elif "online" in name:
        epochs_test.append(epochs)

epochs_train = mne.concatenate_epochs(epochs_train)
epochs_test = mne.concatenate_epochs(epochs_test)

# %%

X_train = epochs_train.get_data(units="uV")
X_test = epochs_test.get_data(units="uV")

y_train = labels_from_epochs(epochs_train)
y_test = labels_from_epochs(epochs_test)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# %%

csp = mne.decoding.CSP(
    n_components=2, reg=None, log=False, transform_into="average_power"
)
lda = BinaryLinearDiscriminantAnalysis(shrinkage="lwf")

model = Pipeline([("csp", csp), ("lda", lda)])

# %%
model.fit(X_train, y_train)

# %%

preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)
print(f"Accuracy: {acc:.2f}")
