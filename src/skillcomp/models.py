"""Multi-label classifiers preserving constant and unseen target columns."""

def estimator(kind, c=1., balanced=False, jobs=4):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import LinearSVC

    if kind == "rf":
        return RandomForestClassifier(random_state=42, n_jobs=jobs)
    class_weight = "balanced" if balanced else None
    if kind == "lr":
        base = LogisticRegression(C=c, class_weight=class_weight, solver="liblinear", max_iter=2000, random_state=42)
    elif kind == "svm":
        base = LinearSVC(C=c, class_weight=class_weight, dual="auto", max_iter=5000, random_state=42)
    else:
        raise ValueError(kind)
    return OneVsRestClassifier(base, n_jobs=jobs)


def continuous_scores(model, x, y_train, kind):
    import numpy as np
    scores = model.decision_function(x) if kind == "svm" else model.predict_proba(x)
    scores = np.asarray(scores)
    all_zero = y_train.sum(axis=0) == 0
    all_one = y_train.sum(axis=0) == len(y_train)
    if kind == "svm":
        scores[:,all_zero] = -np.inf
        scores[:,all_one] = np.inf
    else:
        scores[:,all_zero] = 0.
        scores[:,all_one] = 1.
    return scores

