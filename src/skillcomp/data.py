"""Read-only loading; real database use is restricted to the remote server."""

import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

from .schema import TABLE, TEXT_COLUMNS, label_columns, validate_input_columns


@contextmanager
def open_readonly(path):
    path = Path(path).resolve(strict=True)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as con:
        yield con


def quote_identifier(name):
    return '"' + name.replace('"', '""') + '"'


def load_issue_data(path, input_columns=TEXT_COLUMNS):
    """Return input records and binary targets separately, with explicit names.

    This reader preserves SQL row order; upstream preparation is responsible
    for cleaning and dropping missing text. No split is inferred here.
    """
    with open_readonly(path) as con:
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({quote_identifier(TABLE)})")]
        labels = label_columns(cols)
        inputs = validate_input_columns(input_columns, labels)
        selected = list(inputs) + labels
        sql = f"SELECT {', '.join(map(quote_identifier, selected))} FROM {quote_identifier(TABLE)}"
        rows = con.execute(sql).fetchall()
    x = [dict(zip(inputs, r[:len(inputs)])) for r in rows]
    y = [[int(v is not None and v > 0) for v in r[len(inputs):]] for r in rows]
    return x, y, labels

