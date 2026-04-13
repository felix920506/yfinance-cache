"""
SQLite-based cache backend for yfinance-cache.

Provides SqliteCacheBackend, a drop-in alternative to the file-based
backend in yfc_cache_manager.py.  Each (ticker, object_name) pair maps
to a single row in the cache_data table; serialisation format (pickle
blob vs. JSON text) is chosen by the same rules as the file backend.
"""

import json
import pickle
import sqlite3
import threading
from datetime import datetime, date, timedelta, timezone

import pandas as pd
from zoneinfo import ZoneInfo

from . import yfc_utils as yfcu


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _is_json_serialisable_scalar(obj) -> bool:
    return isinstance(obj, (int, float, str, datetime, date, timedelta))


def _should_use_json(datum) -> bool:
    """Mirror the file-backend logic in GetFilepath for ext selection."""
    if isinstance(datum, list):
        if len(datum) == 0 or isinstance(datum[0], (int, float, str, datetime, date, timedelta)):
            return True
        return False
    if isinstance(datum, dict):
        try:
            json.dumps(datum, default=yfcu.JsonEncodeValue)
            return True
        except (TypeError, OverflowError):
            return False
    if _is_json_serialisable_scalar(datum):
        return True
    return False


def _serialize_datum(datum) -> tuple:
    """Return (blob, json_str) — exactly one is non-None."""
    if _should_use_json(datum):
        return None, json.dumps(datum, default=yfcu.JsonEncodeValue)
    return pickle.dumps(datum, protocol=4), None


def _deserialize_datum(blob, json_str):
    if json_str is not None:
        return json.loads(json_str, object_hook=yfcu.JsonDecodeDict)
    if blob is not None:
        return pickle.loads(blob)
    return None


def _serialize_metadata(md: dict | None) -> str | None:
    if md is None:
        return None
    return json.dumps(md, default=yfcu.JsonEncodeValue)


def _deserialize_metadata(json_str: str | None) -> dict | None:
    if json_str is None:
        return None
    return json.loads(json_str, object_hook=yfcu.JsonDecodeDict)


def _serialize_expiry(expiry: datetime | None) -> str | None:
    if expiry is None:
        return None
    return expiry.isoformat()


def _deserialize_expiry(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.fromisoformat(s)


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_data (
    ticker      TEXT NOT NULL,
    object_name TEXT NOT NULL,
    data_blob   BLOB,
    data_json   TEXT,
    metadata    TEXT,
    expiry      TEXT,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (ticker, object_name)
);

CREATE INDEX IF NOT EXISTS idx_expiry
    ON cache_data(expiry)
    WHERE expiry IS NOT NULL;

CREATE TABLE IF NOT EXISTS cache_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class SqliteCacheBackend:
    """
    SQLite-backed cache with per-thread connections and WAL mode.

    All public methods mirror the file-backend functions in
    yfc_cache_manager so the dispatcher can call them directly.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._local = threading.local()
        self._schema_lock = threading.Lock()
        self._schema_created = False
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        """Return the per-thread connection, creating it lazily."""
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
                timeout=30,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _ensure_schema(self):
        with self._schema_lock:
            if self._schema_created:
                return
            self._conn().executescript(_SCHEMA)
            self._schema_created = True

    def close(self):
        """Close the current thread's connection."""
        conn = getattr(self._local, 'conn', None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def is_datum_cached(self, ticker: str, object_name: str) -> bool:
        row = self._conn().execute(
            "SELECT 1 FROM cache_data WHERE ticker=? AND object_name=?",
            (ticker, object_name),
        ).fetchone()
        return row is not None

    def read_datum(self, ticker: str, object_name: str, return_metadata_too: bool = False):
        row = self._conn().execute(
            "SELECT data_blob, data_json, metadata, expiry "
            "FROM cache_data WHERE ticker=? AND object_name=?",
            (ticker, object_name),
        ).fetchone()

        if row is None:
            return (None, None) if return_metadata_too else None

        # Expiry check
        expiry = _deserialize_expiry(row["expiry"])
        if expiry is not None:
            now = pd.Timestamp.utcnow().replace(tzinfo=ZoneInfo("UTC"))
            if now >= expiry:
                self.delete_datum(ticker, object_name)
                return (None, None) if return_metadata_too else None

        data = _deserialize_datum(row["data_blob"], row["data_json"])
        md = _deserialize_metadata(row["metadata"])

        if expiry is not None:
            if md is None:
                md = {"__expiry__": expiry}
            else:
                md["__expiry__"] = expiry

        return (data, md) if return_metadata_too else data

    def store_datum(
        self,
        ticker: str,
        object_name: str,
        datum,
        expiry: datetime | None = None,
        metadata: dict | None = None,
    ):
        if datum is None:
            self.delete_datum(ticker, object_name)
            return

        # Preserve existing metadata/expiry if caller passes None
        existing = self._conn().execute(
            "SELECT metadata, expiry FROM cache_data WHERE ticker=? AND object_name=?",
            (ticker, object_name),
        ).fetchone()
        if existing is not None:
            if metadata is None:
                metadata = _deserialize_metadata(existing["metadata"])
            if expiry is None:
                expiry = _deserialize_expiry(existing["expiry"])

        blob, json_str = _serialize_datum(datum)
        now_str = datetime.now(timezone.utc).isoformat()

        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                """
                INSERT INTO cache_data
                    (ticker, object_name, data_blob, data_json,
                     metadata, expiry, updated_at)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(ticker, object_name) DO UPDATE SET
                    data_blob  = excluded.data_blob,
                    data_json  = excluded.data_json,
                    metadata   = excluded.metadata,
                    expiry     = excluded.expiry,
                    updated_at = excluded.updated_at
                """,
                (
                    ticker, object_name,
                    blob, json_str,
                    _serialize_metadata(metadata),
                    _serialize_expiry(expiry),
                    now_str,
                ),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def delete_datum(self, ticker: str, object_name: str):
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "DELETE FROM cache_data WHERE ticker=? AND object_name=?",
                (ticker, object_name),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def read_metadata_key(self, ticker: str, object_name: str, key: str):
        row = self._conn().execute(
            "SELECT metadata FROM cache_data WHERE ticker=? AND object_name=?",
            (ticker, object_name),
        ).fetchone()
        if row is None:
            return None
        md = _deserialize_metadata(row["metadata"])
        if md is None:
            return None
        return md.get(key)

    def write_metadata_key(self, ticker: str, object_name: str, key: str, value):
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT metadata FROM cache_data WHERE ticker=? AND object_name=?",
                (ticker, object_name),
            ).fetchone()
            if row is None:
                # Create a placeholder row with null data
                now_str = datetime.now(timezone.utc).isoformat()
                md = {key: value} if value is not None else {}
                conn.execute(
                    """
                    INSERT INTO cache_data
                        (ticker, object_name, pack_name, data_blob, data_json,
                         metadata, expiry, updated_at)
                    VALUES (?,?,NULL,NULL,NULL,?,NULL,?)
                    """,
                    (ticker, object_name, _serialize_metadata(md), now_str),
                )
            else:
                md = _deserialize_metadata(row["metadata"]) or {}
                if value is None:
                    md.pop(key, None)
                else:
                    md[key] = value
                conn.execute(
                    "UPDATE cache_data SET metadata=? WHERE ticker=? AND object_name=?",
                    (_serialize_metadata(md), ticker, object_name),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ------------------------------------------------------------------
    # Ticker enumeration
    # ------------------------------------------------------------------

    def list_tickers(self) -> list:
        rows = self._conn().execute(
            "SELECT DISTINCT ticker FROM cache_data"
        ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Upgrade-flag management (replaces _YFC_/ sentinel files)
    # ------------------------------------------------------------------

    def has_upgrade_flag(self, flag_name: str) -> bool:
        row = self._conn().execute(
            "SELECT 1 FROM cache_meta WHERE key=?",
            (f"_YFC_/{flag_name}",),
        ).fetchone()
        return row is not None

    def set_upgrade_flag(self, flag_name: str):
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "INSERT OR REPLACE INTO cache_meta (key, value) VALUES (?,?)",
                (f"_YFC_/{flag_name}", datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def list_upgrade_flags(self) -> list:
        rows = self._conn().execute(
            "SELECT key FROM cache_meta WHERE key LIKE '_YFC_/%'"
        ).fetchall()
        return [r[0].removeprefix("_YFC_/") for r in rows]

    def delete_upgrade_flag(self, flag_name: str):
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "DELETE FROM cache_meta WHERE key=?",
                (f"_YFC_/{flag_name}",),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
