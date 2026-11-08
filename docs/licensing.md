# License and redistribution review (2026-11-07)

This is a provenance review, not a legal clearance certificate. A content hash
authenticates bytes and does not establish a license or ownership.

| Material | Evidence checked | Candidate handling | Remaining issue |
|---|---|---|---|
| Existing project source | Existing Apache-2.0 LICENSE and NOTICE | Preserve Apache-2.0 and upstream ownership; local portability changes documented | Rightsholders should approve eventual public release |
| Copied NLBSE baseline | Pinned official repository LICENSE, source and README | Original baseline under `upstream/`, with original Apache-2.0 text | Organizer baseline's evaluation TODO remains unresolved |
| Qwen3-Embedding-0.6B and 4B | Each pinned model card explicitly declares `license: apache-2.0` | Cards retained under `third_party/`; no weights in Git | No standalone LICENSE file at either pinned HF revision (HTTP404 recorded); include applicable license/attributions in a future asset release |
| CodeBERT repository code | Official Microsoft CodeBERT repository LICENSE, pinned to fetched repository commit in protocol receipt | MIT copyright/license retained verbatim | Repository code license alone does not establish pinned HF weight redistribution terms |
| microsoft/codebert-base weights | Pinned HF card/API has no license field | Original weight revision and local safetensors conversion hash recorded; no weights in Git | Explicit weight/derivative redistribution confirmation remains required |
| Fine-tuned CodeBERT and final fitted classifier | Frozen project-produced hashes, trained from declared dataset/base model | Listed in separate attachment manifest with null public URL | Inherit applicable upstream/data obligations; release clearance pending, no new license inferred from hash |
| Dataset | Pinned NLBSE/SkillCompetition card/API declares Apache-2.0 | No SQLite/archive/raw issue text distributed | Upstream metadata is not a review of every issue author's rights; retain source provenance and review before redistribution |
| Manuscript | Approved named source/PDF; authors and affiliations retained | Exact copies, separate paper copyright scope | Repository code license does not automatically relicense the paper or transfer rights to a publisher |

See [snapshot receipts](protocol_snapshot.json), [model identities](../assets/attachment_manifest.json),
[CodeBERT MIT text](third_party/codebert_repo_LICENSE), the preserved root `LICENSE`
and `upstream/skill-classification/LICENSE`.

All runtime dependencies remain separately installed under their upstream terms;
their implementation source is not vendored here. The manifest
`third_party_dependencies.json` lists direct dependencies and upstream metadata
sources. It does not claim a complete transitive SBOM or fresh-environment audit.
The pinned safetensors 0.8.0 PyPI metadata endpoint was unavailable in this review;
its historical installed-version record is preserved, but public package
availability and license metadata for that exact build remain to be confirmed.

The old private model ZIP is an integrity-checked handoff. It contains neither the
new named manuscript nor a complete asset licensing/notice dossier and is not
itself an approved public release. Distribute a separately reviewed model attachment
only after the listed obligations are resolved. The current task creates no public URL.
