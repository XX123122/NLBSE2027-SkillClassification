> Historical inference-package scope. Data-root and one-role completion are updated in docs/reproduction_scope.md; this snapshot is not the current completion ledger.

# Training, selection, and reproduction scope

The inference ZIP is deliberately scoped: it runs an existing fitted candidate and does not claim to retrain that candidate or reproduce fivefold development scores. The entire research workspace contains historical runners and pinned configurations. Their presence and cached-run audits do not establish a portable clean rebuild of the latest chain.

The latest dependency endpoints are:

| Stage | Runner in research workspace | Inputs/output scope |
|---|---|---|
| Conditional-union fivefold | `scripts/run_conditional_peer_union_capability.py` | Consumes earlier tree/neighbor/base/publication runs and role-specific caches; produces per-fold training picks, validation locks and held predictions |
| Full-data conditional selection | `scripts/select_full_conditional_peer_union.py` | Consumes previous full deployment and role encoders; fits proper availability centre heads |
| Full-data tree/conditional selection | `scripts/select_full_tree_conditional_union.py` | Fits source-specified SVD/forests; compares on own training OOF |
| Final fit and export | `scripts/export_full_conditional_tree_candidate.py` | Consumes admitted selection and source bundle; fits full final tree/NN/prior and verifies raw inference |
| Audit | `scripts/audit_conditional_peer_union_capability.py`, `scripts/audit_full_tree_conditional_selection.py`, `scripts/audit_full_conditional_tree_export.py` | Independent recomputation and fitted-parameter comparisons; not an independent accuracy dataset |

Before claiming full training reproducibility, a release needs an executable dependency graph starting at the pinned public database and pretrained weights; role-specific preprocessing, labels and components; lexical/base models; supervised CodeBERT checkpoints; auxiliary heads; availability/publication channels; tree/NN selection; threshold locks; and final export. Dynamic paths to historical run caches must be replaced by declared stages or explicitly supplied versioned artifacts. A fresh directory must then execute the chain and independently verify the metrics and roles. Reusing pretrained weights is acceptable; reusing undeclared old fitted caches is not a clean retraining claim.

The current verification instead checks a narrower but necessary property: inference sources relocated to a new directory, original fitted bundle hash unchanged, encoder locations explicitly supplied, and predictions produced without training data or original source imports. These checks do not measure generalization, anonymous public availability, a new installation, or a new training run.
