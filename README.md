# NLBSE 2027 Skill Classification

This is the **public source repository** for the skill tool at https://github.com/XX123122/NLBSE2027-SkillClassification, made public with owner authorization on 2026-11-16. Frozen inference assets are available in the [Release](https://github.com/XX123122/NLBSE2027-SkillClassification/releases/tag/v0.1.0-private). Hosting does not establish official qualification, a complete training rebuild, or new upstream redistribution rights. Experimental limitations and license review items remain explicit.

Xin Xu is first author; Jun Yu is second and corresponding author, University of
Science and Technology of China. The paper preserves both approved affiliations
and email addresses. Use [the named paper](paper/skill_classification_named.pdf)
and [the matching revised source](paper/main.tex). This is the Nov.16 public-hosting revision (same experimental results as Nov.8), with baseline, ablation, tail-label and unsuccessful weight-edit evidence. Earlier private handoff PDFs are historical.

## Scope and layout

- `src/`: runtime models and methods; `scripts/`: public CLIs plus historical producers.
- `configs/`: portable role recipes and explicitly marked historical configuration snapshots.
- `tests/`: synthetic fixtures only; `docs/`: reproduction scope, source graphs and licensing.
- `notebooks/`: scope note, with no executed corpus notebooks.
- `paper/`: the single named four-page Skill Classification manuscript and source.
- `results/`: saved aggregate results/evidence, with no row predictions or raw issue text.
- `assets/`: attachment manifests only. Weight files, encoders, data and caches are ignored by Git.

**Runnable inference and full training reproduction are different deliverables.**
Historical isolated inference was verified; the data-root generation and one fixed
`fold0_inner0` producer chain were independently rebuilt. Other roles, the complete
selection/calibration/evaluation chain and a complete clean export rebuild have not
all been re-executed. The new role adapter has synthetic/static validation only.
See [scope](docs/reproduction_scope.md), [machine-readable gaps](docs/reproduction_gaps.json)
and [source graph](docs/source_graph/pipeline.json).

## Install

Real inference/training targets Linux/CUDA. CPU and encoder runtimes are separate;
the observed historical environments were reused, not freshly installed here.
Python 3.11 is used for the CPU classifier and Python 3.12.3 for the encoder runtime.
Set up the two environments separately; a compatible CUDA PyTorch wheel is required.

```sh
python3.11 -m venv .venv-cpu
.venv-cpu/bin/python -m pip install -r requirements-cpu.txt -r requirements-tests.txt
.venv-cpu/bin/python -m pip install --no-deps -e .
python3.12 -m venv .venv-encoder
# Install a torch==2.10.0 build compatible with the local CUDA driver first.
# Historical build: 2.10.0+cu130; requirements-encoder.txt does not select that wheel index.
.venv-encoder/bin/python -m pip install -r requirements-encoder.txt
.venv-encoder/bin/python -m pip install --no-deps -e .
export SKILLCOMP_ENCODER_PYTHON="$PWD/.venv-encoder/bin/python"
export PYTHONPATH="$PWD/src:$PWD/scripts"
```

The CPU environment uses NumPy 1.26.4; the historical encoder environment used
NumPy 2.5.2. Do not combine both requirement files. Full inference peak GPU memory
is unmeasured; **11,000 MiB is the worker reservation, not a measured peak**.
The historical one-role CodeBERT training reported about 3.763 GiB peak allocated
memory. See [runtime requirements](assets/runtime_requirements.json).

## Test without real models or data

```sh
.venv-cpu/bin/python -m pytest -q tests
.venv-cpu/bin/python scripts/check_saved_results.py
.venv-cpu/bin/python scripts/reproduce.py --role fold0_inner0
.venv-cpu/bin/python scripts/reproduce.py --role full_final
.venv-cpu/bin/python scripts/verify_repository.py
```

These commands use synthetic fixtures, read source/aggregate records or emit plans.
They do not load real weights, query SQLite, encode real text, fit models, search
thresholds or compute new data scores. Tests can also run on Windows with a compatible
CPU environment. See [validation](docs/validation.md) for what was actually tested.

## Obtain assets

The frozen classifier and its required encoders/checkpoint are hosted separately in
the [public Release](https://github.com/XX123122/NLBSE2027-SkillClassification/releases/tag/v0.1.0-private). No repository authorization is required.
The complete asset-only ZIP exceeds GitHub's per-file limit, so it is distributed
as byte fragments below 2 GiB each, with per-part and reassembled SHA256 values.
Use [the release index](assets/private_release_manifest.json), download every part
to one folder, then verify/reassemble and extract explicitly:

```sh
python scripts/reassemble_release_assets.py --parts-dir /path/to/downloads --output /path/to/skill-frozen-assets.zip
python scripts/manage_assets.py --archive /path/to/skill-frozen-assets.zip --manifest assets/private_release_manifest.json --extract-to assets
cp configs/asset-map.example.json asset-map.json
```

The new asset-only payload contains only frozen model/encoder bytes, their manifests
and applicable provenance/license notes. No training corpus, database, raw issues,
experimental edited checkpoint or historical nested code ZIP is uploaded. Original
model SHA256 is `44e367a74fd23c1a5538f655c18238fc6e085a6f9a96415552e8a08a2bb02acf`;
the fine-tuned CodeBERT checkpoint is
`469a51b9e07484e2aa9f258581b9d32ed080dd0762962016cfe17560a481d7a1`.
CodeBERT's base safetensors is a local format conversion, not an upstream original.
Public hosting does not resolve upstream redistribution obligations; review
[licensing](docs/licensing.md). Existing legacy ZIPs remain usable with the original
attachment manifest. Never disable frozen-bundle hash authentication.

For later authorized training, acquire the official dataset separately from
[the pinned upstream snapshot](https://huggingface.co/datasets/NLBSE/SkillCompetition/tree/7f3ea04c65c669c08a46d2ad11d610a099737b08).
For example, with the separately installed HF CLI:

```sh
hf download NLBSE/SkillCompetition skillscope_data.zip --repo-type dataset \
  --revision 7f3ea04c65c669c08a46d2ad11d610a099737b08 --local-dir data/raw
# Extract skillscope_data.db from the downloaded ZIP into data/raw/.
```

The required SQLite SHA256 is
`ed47eb3c2173eb137c6146df78e750c261ff2a9f42f631c8e0186fa4c94022ea`.
The reproduction runner verifies it before use. Review [license evidence and gaps](docs/licensing.md)
before distributing source, data-derived fitted assets or upstream weights.

## Inference (future local real execution, not performed in this packaging task)

After assets are supplied, explicitly enable Linux inference and configure the local
GPU exclusions in `configs/gpu_resource_policy.json`. Shared hosts should use a common
`SKILLCOMP_GPU_REGISTRY_ROOT`; reserve devices according to the host policy.

```sh
mkdir -p outputs cache
touch .skillcomp-server
.venv-cpu/bin/python scripts/predict_competition_reserved.py \
  --bundle assets/model.joblib --asset-map asset-map.json --check-only
.venv-cpu/bin/python scripts/predict_competition_reserved.py \
  --bundle assets/model.joblib --asset-map asset-map.json \
  --input examples/inputs.json --output outputs/predictions.json
```

Input is exactly a JSON array of objects with **three string fields**:
`issue text`, `issue description`, `Repo Name`. Extra fields (query labels, PR IDs,
patches, database paths) are rejected. Output includes the authenticated bundle's
unchanged ordered 217 `label_order` names; each prediction has a 217-bit `vector`
and selected label names. Empty lists produce no predictions and no GPU work;
blank strings are valid inputs; unknown repositories use fitted global fallbacks.
No runtime database or query labels are required. The original bundle hash is
checked before deserialization; encoder paths are rebound only in memory.

## Training, tuning and evaluation entrypoints

The role runner defaults to a **plan** and accepts 24 historical role names:
five development folds times three inner roles plus one validation role, and four
full-data roles. It never invents an official split.

```sh
python scripts/reproduce.py --role fold0_inner0 --stage prepare
python scripts/reproduce.py --role fold0_inner0 --phase tune
python scripts/reproduce.py --role fold0_inner0 --phase evaluate
python scripts/reproduce.py --role full_final --phase export
```

The latter three commands report the source-only/unconnected phases and their
blockers. Adding `--execute` to a blocked phase fails closed. They are accurate
entrypoints to the **remaining work**, not claims of a complete working training
pipeline. Historical tuners, evaluators and exporters are included for review in
`scripts/`, with their configuration graph in `configs/historical/` and
`docs/source_graph/historical_run_graph.json`. They require missing dynamic inputs
and existing producer outputs; running them directly is not a clean reproduction.

A future authorized fixed-role training sequence is documented in
[training commands](docs/training_commands.md). The public runner can create an
isolated local work directory, regenerate data/roles, tokenize, train/extract one
development encoder and fit the fixed base head. The active tree/neighbor producer
is connected for `fold0_inner0` only. Its original-whitespace base branch and the
selected normalized-whitespace branch are kept distinct. Full-data four-epoch
execution remains blocked until its exact historical schedule is resolved; the
plan reports this instead of guessing it. New adapter code has not been run on
real data in this task.

## Results and limitations

| Internal component-grouped fivefold method | Precision | Recall | micro-F1 |
|---|---:|---:|---:|
| Matched RF | 0.6854670283 | 0.4848024632 | 0.5679309017 |
| Previous tree/neighbor strategy | 0.6861776173 | 0.5558200012 | 0.6141577704 |
| Training-selected conditional strategy | 0.6860585657 | 0.5562402480 | 0.6143664995 |

System F1 gain over matched RF: **4.64356 percentage points**. Last-step gain over
the strong prior strategy: **0.02087 points**. Precision margin over RF is only
**0.05915 points**, with descriptive component/repository intervals crossing zero.
These are retrospective, repeatedly exposed weak-label development results.
Shared encoder/projection centre approximations remain. The full-data frozen model
used a separate training OOF choice, F1 **0.6095585841**, and has **no independent test score**.

Official train/test assignment, 217/142 label scoring and final reference values are
still unresolved in [the organizer TODO](https://github.com/nlbse2027/skill-classification/blob/main/competition_baseline/TODO.md),
checked 2026-11-07. Formal evaluation must retrain on permitted training roles; the
all-data fitted bundle must not be tested on overlapping training rows. Diagnostic
target reconstruction never replaces the organizer labels automatically.

## Submission boundary

Only **Skill Classification** belongs to this candidate. The confirmed form is
[NLBSE'27 Tool Submission](https://forms.gle/yx2CcKLYYBSjWoiJ8); one entry selects one
competition and one PDF. The tool deadline is 2026-11-27 AoE, equivalent to
2026-11-28 19:59:59 Asia/Shanghai. The tool submission need not be double-blind under
the user-confirmed instructions. Recheck the official site before actual submission;
do not substitute the general workshop deadline.

See [release gaps](docs/reproduction_gaps.json). Source and model hosting, final
protocol verification, weight redistribution review and complete training closure
remain distinct requirements. No form was opened or filled, and no organizer was contacted. Experiments are stopped; this private upload runs no training, real-model forward, threshold search or new data evaluation.

## Submission preparation recheck (2026-11-16)

The named four-page PDF now includes the real repository and private Release location. All result tables and authors are unchanged. The organizer TODO still leaves split/label scope/reference values unresolved; see [dated source check](docs/submission_protocol_check_20261116.json). The owner authorized public access on Nov.16. The source and existing model Release are public; no competition form has been submitted.

## Public access (2026-11-16)

The model Release retains the legacy `v0.1.0-private` tag only to preserve stable URLs; repository visibility controls access and is now public. Use [the current public index](assets/public_release_manifest.json). The old private index Release attachment is a historical record; all model part bytes/hashes are unchanged. The Apache-2.0 source license and original notices remain. Read [licensing caveats](docs/licensing.md), including unresolved pinned CodeBERT weight/derivative redistribution terms.
