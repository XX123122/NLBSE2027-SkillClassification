"""Input allowlist verified against upstream competition_baseline/main.py.

The label list must be derived from the actual remote database, not hardcoded.
Metadata is used for identity/auditing only and is excluded from model inputs.
"""

TABLE = "nlbse_tool_competition_data_by_issue"
TEXT_COLUMNS = ("issue text", "issue description")
METADATA_COLUMNS = (
    "Repo Name", "PR #", *TEXT_COLUMNS, "created_at", "closed_at",
    "userlogin", "author_name", "most_recent_commit",
)


def label_columns(columns):
    columns = list(columns)
    if columns[:len(METADATA_COLUMNS)] != list(METADATA_COLUMNS):
        raise ValueError("Schema differs from the audited upstream metadata prefix")
    labels = columns[len(METADATA_COLUMNS):]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("Empty or duplicate label schema")
    return labels


def validate_input_columns(columns, labels):
    columns = tuple(columns)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Input columns must be nonempty and unique")
    if set(columns) & set(labels) or not set(columns) <= set(TEXT_COLUMNS):
        raise ValueError("Only audited issue title/description may enter model inputs")
    return columns


def validate_metadata_inputs(columns, labels, allow_repo_metadata=False):
    """Explicit opt-in for Repo Name, permitted by official issue-view comments.

    PR identity, dates, authors and all target-derived fields remain excluded.
    The initial text-only reader retains its stricter default allowlist.
    """
    columns = tuple(columns)
    allowed = set(TEXT_COLUMNS) | ({"Repo Name"} if allow_repo_metadata else set())
    if not columns or len(columns)!=len(set(columns)) or not set(columns)<=allowed or set(columns)&set(labels):
        raise ValueError("Unapproved metadata or target column in inputs")
    return columns

