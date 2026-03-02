import numpy as np


def _markers_from_events(events, event_id):
    event_desc = {v: k for k, v in event_id.items()}

    samples = np.array(events)[:, 0]

    markers = list()
    for val in np.array(events)[:, 2]:
        markers.append(str(event_desc[val]))

    return samples, markers


def labels_from_epochs(epochs, mappings=None):
    y = list()

    _, markers = _markers_from_events(epochs.events, epochs.event_id)

    if mappings is None:
        return np.array(markers)

    for marker in markers:
        for key, val in mappings.items():
            if "/" in marker:
                if key in marker.split("/"):
                    y.append(val)
            else:
                if key in marker:
                    y.append(val)

    if len(epochs) != len(y):
        raise RuntimeError(
            f"lenth of epochs is not match with length of y.\n len(epochs): {len(epochs)}, len(y): {len(y)}"
        )

    return np.array(y)
