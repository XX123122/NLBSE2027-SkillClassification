# Validation scope

This packaging task permits synthetic/static/saved-result checks only. It performs
zero real encoder/model forwards, training, threshold search, new corpus scoring
or remote tasks. The input model ZIP is streamed for SHA256 and CRC verification;
its joblib/checkpoint files are never deserialized. Models and datasets are not
extracted into this source tree.

The source ZIP must be unpacked into an independent migration directory, and the
synthetic tests, saved aggregate check, source/link audit, role plans and file
manifest check must run against that copy. `results/local_validation.json` records
the final results. A stdlib audit guard blocks model/data reads, network and GPU
tools during that validation. A complete fresh dependency install is not claimed.

Repository-relative Markdown links are checked for existing targets and anchors.
Remote links remain upstream references and are not treated as internal files.
Official/model/data source pages actually retrieved for this review have dated
content receipts. All Python files are parsed/compiled without executing their
real-data stages; imports and literal worker references have a source graph.

The named manuscript is copied byte-for-byte, checked for four pages and approved
authors/emails/corresponding footnote, and visually inspected. No manuscript text,
weights, threshold or old review archive is overwritten.
