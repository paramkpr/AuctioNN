"""
results_logger.py
Fast, append-only result sink.

Rows are buffered in memory and flushed to a single Parquet file every
`flush_every` impressions.  Use the same schema throughout the run, so
PyArrow can stream-append efficiently.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class ResultsLogger:
    def __init__(
        self,
        out_path: str | Path,
        flush_every: int = 10_000,
    ) -> None:
        self._out_path = Path(out_path)
        self._flush_every = flush_every
        self._buffer: List[Mapping[str, Any]] = []
        self._writer: pq.ParquetWriter | None = None

    # ── public ──────────────────────────────────────────────────────
    def log(self, row: Mapping[str, Any]) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return

        # Convert buffered rows → Arrow Table
        df = pd.DataFrame(self._buffer)
        table = pa.Table.from_pandas(df, preserve_index=False)

        # First flush → create writer
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._out_path, table.schema)
        self._writer.write_table(table)

        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        if self._writer:
            self._writer.close()
            self._writer = None

    # Defensive destructor — just in case
    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
