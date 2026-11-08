# Reproduction evidence: completed, source-only and unverified

The latest evidence supersedes earlier pending notes for the data-root and fixed
role only. It does not turn the old inference ZIP into a complete training package.

| Stage | Historical independent execution | Candidate source status | This packaging round |
|---|---|---|---|
| Published dataset, diagnostic targets, components, five internal roles | Rebuilt from pinned read-only SQLite, array identities matched | `isolated_data_root.py` | Source/receipt check only |
| CodeBERT supervision, lexical/base head for fold0_inner0 | 201 checkpoint tensors, losses, learned vectors and base scores matched | Fixed producer plus parameterized `prepare_role.py` | Synthetic/static adapter tests only |
| Selected normalized branch and active tree/conditional-NN policy, fold0_inner0 | 1024 tree parameters and fitted heads matched; max score difference 3.33e-16 | `competition_active_role_worker.py`, bounded runner | No real execution |
| Frozen Qwen4 input-only transform in that continuation | Existing cache explicitly declared and hash/text identity checked; not re-encoded | Cache absent; provenance requirement explicit | No re-encoding |
| Remaining development roles | Existing fitted artifacts/lineage verified, not complete independent rebuild | Source and 24-role plan; producer adapters unverified numerically | Synthetic role checks only |
| Full-data four-epoch roles | Existing full model/export component checks; no complete root-to-export rebuild | Historical source/config; schedule execution blocked | No real execution |
| All candidate/reference OOF selection, validation thresholds, held scoring | Original locked historical run verified | Historical source only; missing dynamic inputs | Saved aggregate consistency only |
| Full export | Existing bundle, parameter refits and raw/cache equivalence verified | Source included; complete producer closure missing | Hash verification only |
| Raw inference migration | Historical 8 raw + 2 synthetic inputs, known/unseen repo, empty/invalid handling, original corpus/cache/network blocked | Portable inference entrypoint | No real model/encoder forward; synthetic tests only |

The 51-node historical run graph and source import graph have different meanings.
`historical_run_graph.json` describes recorded run/config dependencies; it is not an
executable full producer closure. `python_dependencies.json` describes source
imports and worker filename references. `pipeline.json` connects stage contracts
and evidence status. Missing assets and unresolved stages are enumerated in
`../docs/reproduction_gaps.json` and `source_graph/required_historical_inputs.json`.

The new portable role runner derives 24 named roles without assuming an official
split. It connects the original single-role stage interfaces and parameterizes
base preparation. The adapter and site-neutral scheduler edits were tested only
with synthetic/static inputs. Historical numerical evidence is retained as
historical evidence, never attributed to a real execution of this newly packaged
source tree.

The public interface rejects extra fields and validates the 217-label output
invariant. Label identity/order is taken from the authenticated fixed bundle; no
new label ordering is manufactured from the diagnostic targets. The full-data
bundle is deployment state and cannot establish an independent score on its own
training rows.
