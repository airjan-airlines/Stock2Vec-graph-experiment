from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lancedb
import numpy as np
import pyarrow as pa

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "data" / "vectors" / "stock2vec.lancedb"

VECTOR_COLUMN = "vector"
DEFAULT_METRIC = "cosine"


def normalize_vector(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v.astype(np.float32) / n if n > 0 else v.astype(np.float32)


def db_path_for(daily: bool = False) -> Path:
    name = "stock2vec_daily.lancedb" if daily else "stock2vec.lancedb"
    return BASE_DIR / "data" / "vectors" / name


class VectorStore:
    def __init__(self, db_path: str | Path | None = None, table_name: str = "embeddings"):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.table_name = table_name
        self._db: lancedb.LanceDBConnection | None = None
        self._table: lancedb.table.LanceTable | None = None

    def connect(self) -> lancedb.LanceDBConnection:
        if self._db is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.db_path))
        return self._db

    @property
    def table(self):
        if self._table is None:
            db = self.connect()
            try:
                self._table = db.open_table(self.table_name)
            except Exception:
                raise FileNotFoundError(
                    f"Table '{self.table_name}' not found in {self.db_path}. "
                    "Create it first with create_table()."
                )
        return self._table

    def _infer_schema(self, record: dict, embedding_dim: int) -> pa.Schema:
        fields = [
            pa.field(VECTOR_COLUMN, pa.list_(pa.float32(), embedding_dim)),
            pa.field("ticker", pa.string()),
            pa.field("timestamp", pa.string()),
            pa.field("price", pa.float64()),
            pa.field("realized_vol", pa.float64()),
            pa.field("fwd_ret_30min", pa.float64()),
            pa.field("fwd_ret_1d", pa.float64()),
            pa.field("fwd_ret_5d", pa.float64()),
            pa.field("regime", pa.string()),
        ]
        extra_cols = [k for k in record if k not in {f.name for f in fields}]
        for c in extra_cols:
            val = record[c]
            if isinstance(val, (int, float, np.floating, np.integer)):
                fields.append(pa.field(c, pa.float64()))
            elif isinstance(val, str):
                fields.append(pa.field(c, pa.string()))
            elif isinstance(val, np.ndarray):
                fields.append(pa.field(c, pa.list_(pa.float32(), val.shape[0])))
        return pa.schema(fields)

    def create_table(self, records: list[dict], embedding_dim: int,
                     replace: bool = False, build_index: bool = True) -> None:
        db = self.connect()
        schema = self._infer_schema(records[0], embedding_dim) if records else None

        if replace:
            db.drop_table(self.table_name, ignore_missing=True)

        tbl = db.create_table(self.table_name, data=records, schema=schema,
                               mode="overwrite" if replace else "create")
        if build_index:
            tbl.create_index(metric=DEFAULT_METRIC, num_partitions=256, num_sub_vectors=16)
            log.info(f"Built IVF-PQ index on '{self.table_name}' (metric={DEFAULT_METRIC})")
        self._table = tbl
        log.info(f"Table '{self.table_name}' created at {self.db_path}  ({len(records)} rows)")

    def insert(self, records: list[dict]) -> None:
        self.table.add(records)
        log.info(f"Inserted {len(records)} records into '{self.table_name}'")

    def search(self, vector: np.ndarray, k: int = 10) -> list[dict[str, Any]]:
        query = normalize_vector(vector).reshape(1, -1).tolist()[0]
        result = self.table.search(query, vector_column_name=VECTOR_COLUMN).limit(k).to_list()
        return result

    def row_count(self) -> int:
        return self.table.count_rows()

    def close(self) -> None:
        self._table = None
        self._db = None
