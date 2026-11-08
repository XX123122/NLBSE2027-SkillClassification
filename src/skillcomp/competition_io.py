"""Public output invariant; label identity comes from the authenticated frozen bundle."""


def validate_label_order(labels):
    if (not isinstance(labels, list) or len(labels) != 217 or
            any(not isinstance(name, str) or not name for name in labels) or len(set(labels)) != 217):
        raise ValueError('Frozen bundle must supply exactly 217 distinct ordered label names')
    return labels
