"""Public skill-name parent relations, independent of any target values."""


def parents_from_labels(labels):
    if len(set(labels))!=len(labels):raise ValueError('Unique label names required')
    lookup={label:index for index,label in enumerate(labels)}
    parents=[lookup.get(label.split('-',1)[0],-1) if '-' in label else -1 for label in labels]
    if any(parent>=0 and parents[parent]>=0 for parent in parents):raise ValueError('One-level taxonomy required')
    return parents
