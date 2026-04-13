"""
SQLite-based cache backend for yfinance-cache.

Structured objects (price history, dividends, splits, earnings dates) are
stored in proper relational tables with typed columns.  Everything else
(info, calendar, financials, …) goes into a catch-all KV table.

Public API is identical to the old blob-only version so no changes are
needed in yfc_cache_manager.py.
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
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_kv (
    ticker      TEXT NOT NULL,
    object_name TEXT NOT NULL,
    data_blob   BLOB,
    data_json   TEXT,
    PRIMARY KEY (ticker, object_name)
);

CREATE TABLE IF NOT EXISTS object_metadata (
    ticker        TEXT NOT NULL,
    object_name   TEXT NOT NULL,
    metadata_json TEXT,
    expiry_ns     INTEGER,
    tz_name       TEXT,
    PRIMARY KEY (ticker, object_name)
);

CREATE TABLE IF NOT EXISTS price_history (
    ticker            TEXT    NOT NULL,
    interval          TEXT    NOT NULL,
    dt_ns             INTEGER NOT NULL,
    open              REAL,
    high              REAL,
    low               REAL,
    close             REAL,
    volume            REAL,
    dividends         REAL,
    splits            REAL,
    fetch_date_ns     INTEGER,
    final             INTEGER,
    c_check           INTEGER,
    csf               REAL,
    cdf               REAL,
    last_div_adj_ns   INTEGER,
    last_split_adj_ns INTEGER,
    repaired          INTEGER,
    PRIMARY KEY (ticker, interval, dt_ns)
);

CREATE TABLE IF NOT EXISTS dividends (
    ticker              TEXT    NOT NULL,
    dt_ns               INTEGER NOT NULL,
    amount              REAL,
    fetch_date_ns       INTEGER,
    close_before        REAL,
    close_repaired      INTEGER,
    back_adj            REAL,
    superseded_div      REAL,
    superseded_back_adj REAL,
    superseded_fetch_ns INTEGER,
    PRIMARY KEY (ticker, dt_ns)
);

CREATE TABLE IF NOT EXISTS splits (
    ticker              TEXT    NOT NULL,
    dt_ns               INTEGER NOT NULL,
    ratio               REAL,
    fetch_date_ns       INTEGER,
    superseded_split    REAL,
    superseded_fetch_ns INTEGER,
    PRIMARY KEY (ticker, dt_ns)
);

CREATE TABLE IF NOT EXISTS earnings_dates (
    ticker         TEXT    NOT NULL,
    dt_ns          INTEGER NOT NULL,
    reported_eps   REAL,
    expected_eps   REAL,
    surprise_pct   REAL,
    event_type     TEXT,
    fetch_date_ns  INTEGER,
    date_confirmed INTEGER,
    PRIMARY KEY (ticker, dt_ns)
);

CREATE TABLE IF NOT EXISTS cache_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

_STRUCTURED = frozenset({'dividends', 'splits', 'earnings_dates'})


def _route(object_name: str) -> tuple:
    """Return (table_name, extra).  extra is the interval for price_history."""
    if object_name.startswith('history-'):
        return 'price_history', object_name[8:]
    if object_name in _STRUCTURED:
        return object_name, None
    return 'cache_kv', None


# ---------------------------------------------------------------------------
# Datetime helpers  (all datetimes stored as UTC nanoseconds INTEGER)
# ---------------------------------------------------------------------------

def _dt_to_ns(ts) -> int | None:
    """Timezone-aware Timestamp/datetime → UTC nanoseconds. None for NaT/None."""
    if ts is None:
        return None
    try:
        ts = pd.Timestamp(ts)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return int(ts.value)   # .value is always UTC ns


def _ns_to_ts(ns: int | None, tz=None) -> pd.Timestamp | None:
    """UTC ns integer → Timestamp, optionally tz-converted."""
    if ns is None:
        return None
    ts = pd.Timestamp(ns, unit='ns', tz='UTC')
    if tz is not None:
        ts = ts.tz_convert(tz)
    return ts


def _bool_to_int(v) -> int | None:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return int(bool(v))


def _int_to_bool(v):
    """Convert a SQLite INTEGER (0/1/NULL) to a Python bool or pd.NA."""
    return pd.NA if v is None else bool(v)


def _real(v) -> float | None:
    """Scalar → float, NaN → None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Column-array builders (vectorised helpers for DataFrame → SQL)
# ---------------------------------------------------------------------------

def _col_real(df: pd.DataFrame, name: str) -> list:
    if name not in df.columns:
        return [None] * len(df)
    return [_real(v) for v in df[name]]


def _col_dt_ns(df: pd.DataFrame, name: str) -> list:
    if name not in df.columns:
        return [None] * len(df)
    return [_dt_to_ns(v) for v in df[name]]


def _col_bool(df: pd.DataFrame, name: str) -> list:
    if name not in df.columns:
        return [None] * len(df)
    return [_bool_to_int(v) for v in df[name]]


def _col_text(df: pd.DataFrame, name: str) -> list:
    if name not in df.columns:
        return [None] * len(df)
    return [None if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)
            for v in df[name]]


# ---------------------------------------------------------------------------
# KV serialisation helpers (JSON or pickle blob)
# ---------------------------------------------------------------------------

def _should_use_json(datum) -> bool:
    if isinstance(datum, list):
        # Check the whole list, not just the first element.  Checking only
        # datum[0] would miss later elements with non-serialisable types and
        # cause json.dumps() to raise inside _serialize_kv with no fallback.
        if len(datum) == 0:
            return True
        try:
            json.dumps(datum, default=yfcu.JsonEncodeValue)
            return True
        except (TypeError, OverflowError):
            return False
    if isinstance(datum, dict):
        try:
            json.dumps(datum, default=yfcu.JsonEncodeValue)
            return True
        except (TypeError, OverflowError):
            return False
    return isinstance(datum, (int, float, str, datetime, date, timedelta))


def _serialize_kv(datum) -> tuple:
    """Return (blob, json_str) — exactly one is non-None."""
    if _should_use_json(datum):
        return None, json.dumps(datum, default=yfcu.JsonEncodeValue)
    return pickle.dumps(datum, protocol=4), None


def _deserialize_kv(blob, json_str):
    if json_str is not None:
        return json.loads(json_str, object_hook=yfcu.JsonDecodeDict)
    if blob is not None:
        return pickle.loads(blob)
    return None


def _serialize_metadata(md: dict | None) -> str | None:
    if not md:
        return None
    return json.dumps(md, default=yfcu.JsonEncodeValue)


def _deserialize_metadata(s: str | None) -> dict | None:
    if not s:
        return None
    return json.loads(s, object_hook=yfcu.JsonDecodeDict)


def _serialize_expiry(expiry) -> int | None:
    return _dt_to_ns(expiry)


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class SqliteCacheBackend:
    """
    SQLite-backed cache with per-thread connections and WAL mode.

    Structured objects use typed tables; everything else uses cache_kv.

    Each thread gets its own sqlite3.Connection (via threading.local) so
    that WAL-mode concurrent reads don't contend with each other.  All
    connections are tracked in ``_all_conns`` so that ``close()`` can
    release every file descriptor regardless of which thread calls it.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._local = threading.local()
        self._schema_lock = threading.Lock()
        self._schema_created = False
        # Track every connection created across all threads so close() can
        # flush and shut them all down, preventing file-descriptor leaks when
        # worker threads terminate without calling close() themselves.
        self._all_conns: list[sqlite3.Connection] = []
        self._all_conns_lock = threading.Lock()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
                timeout=30,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
            with self._all_conns_lock:
                self._all_conns.append(conn)
        return conn

    def _ensure_schema(self):
        with self._schema_lock:
            if self._schema_created:
                return
            self._conn().executescript(_SCHEMA)
            self._schema_created = True

    def close(self):
        """Close all connections opened by any thread and checkpoint the WAL.

        Safe to call from any thread.  After this call the backend should not
        be used further (each thread that needs it again must re-open via a
        new SqliteCacheBackend instance).
        """
        with self._all_conns_lock:
            for conn in self._all_conns:
                try:
                    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except Exception:
                    pass
                try:
                    conn.close()
                except Exception:
                    pass
            self._all_conns.clear()
        # Also clear the current thread's reference so _conn() re-creates if needed.
        self._local.conn = None

    # ------------------------------------------------------------------
    # Internal: metadata table
    # ------------------------------------------------------------------

    def _read_object_metadata(self, ticker: str, object_name: str):
        """Return (tz_name, metadata_dict, expiry_ns)."""
        row = self._conn().execute(
            "SELECT metadata_json, expiry_ns, tz_name "
            "FROM object_metadata WHERE ticker=? AND object_name=?",
            (ticker, object_name),
        ).fetchone()
        if row is None:
            return None, None, None
        return row['tz_name'], _deserialize_metadata(row['metadata_json']), row['expiry_ns']

    def _upsert_object_metadata(self, conn, ticker, object_name,
                                 metadata, expiry, tz_name):
        """Write metadata, preserving existing values for None args."""
        existing = conn.execute(
            "SELECT metadata_json, expiry_ns, tz_name "
            "FROM object_metadata WHERE ticker=? AND object_name=?",
            (ticker, object_name),
        ).fetchone()

        if existing is not None:
            md_json  = _serialize_metadata(metadata) if metadata is not None \
                       else existing['metadata_json']
            exp_ns   = _serialize_expiry(expiry) if expiry is not None \
                       else existing['expiry_ns']
            tz_final = tz_name if tz_name is not None else existing['tz_name']
        else:
            md_json  = _serialize_metadata(metadata)
            exp_ns   = _serialize_expiry(expiry)
            tz_final = tz_name

        conn.execute(
            """INSERT OR REPLACE INTO object_metadata
                (ticker, object_name, metadata_json, expiry_ns, tz_name)
               VALUES (?,?,?,?,?)""",
            (ticker, object_name, md_json, exp_ns, tz_final),
        )

    def _check_and_delete_if_expired(self, ticker: str, object_name: str) -> bool:
        """Return True if the KV item has expired (and atomically deletes it).

        The check and delete happen inside a single BEGIN IMMEDIATE transaction
        to prevent a TOCTOU race where another thread writes a fresh value
        between this thread's expiry check and the subsequent delete.
        """
        conn = self._conn()
        now_ns = int(pd.Timestamp.now('UTC').value)
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT expiry_ns FROM object_metadata WHERE ticker=? AND object_name=?",
                (ticker, object_name),
            ).fetchone()
            if row is None or row['expiry_ns'] is None or now_ns < row['expiry_ns']:
                conn.rollback()
                return False
            # Expired: delete data and metadata atomically within this transaction.
            conn.execute(
                "DELETE FROM cache_kv WHERE ticker=? AND object_name=?",
                (ticker, object_name),
            )
            conn.execute(
                "DELETE FROM object_metadata WHERE ticker=? AND object_name=?",
                (ticker, object_name),
            )
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise

    # ------------------------------------------------------------------
    # Internal: structured table writers
    # ------------------------------------------------------------------

    def _store_price_history(self, conn, ticker: str, interval: str, df: pd.DataFrame):
        conn.execute(
            "DELETE FROM price_history WHERE ticker=? AND interval=?",
            (ticker, interval),
        )
        if df.empty:
            return

        n = len(df)
        dt_ns = df.index.asi8.tolist()   # UTC ns from DatetimeIndex (always UTC internally)

        rows = list(zip(
            [ticker] * n, [interval] * n, dt_ns,
            _col_real(df, 'Open'),
            _col_real(df, 'High'),
            _col_real(df, 'Low'),
            _col_real(df, 'Close'),
            _col_real(df, 'Volume'),
            _col_real(df, 'Dividends'),
            _col_real(df, 'Stock Splits'),
            _col_dt_ns(df, 'FetchDate'),
            _col_bool(df, 'Final?'),
            _col_bool(df, 'C-Check?'),
            _col_real(df, 'CSF'),
            _col_real(df, 'CDF'),
            _col_dt_ns(df, 'LastDivAdjustDt'),
            _col_dt_ns(df, 'LastSplitAdjustDt'),
            _col_bool(df, 'Repaired?'),
        ))

        conn.executemany(
            """INSERT INTO price_history
                (ticker, interval, dt_ns, open, high, low, close, volume,
                 dividends, splits, fetch_date_ns, final, c_check, csf, cdf,
                 last_div_adj_ns, last_split_adj_ns, repaired)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )

    def _store_dividends(self, conn, ticker: str, df: pd.DataFrame):
        conn.execute("DELETE FROM dividends WHERE ticker=?", (ticker,))
        if df.empty:
            return
        n = len(df)
        rows = list(zip(
            [ticker] * n, df.index.asi8.tolist(),
            _col_real(df, 'Dividends'),
            _col_dt_ns(df, 'FetchDate'),
            _col_real(df, 'Close before'),
            _col_bool(df, 'Close repaired?'),
            _col_real(df, 'Back Adj.'),
            _col_real(df, 'Superseded div'),
            _col_real(df, 'Superseded back adj.'),
            _col_dt_ns(df, 'Superseded div FetchDate'),
        ))
        conn.executemany(
            """INSERT INTO dividends
                (ticker, dt_ns, amount, fetch_date_ns, close_before, close_repaired,
                 back_adj, superseded_div, superseded_back_adj, superseded_fetch_ns)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )

    def _store_splits(self, conn, ticker: str, df: pd.DataFrame):
        conn.execute("DELETE FROM splits WHERE ticker=?", (ticker,))
        if df.empty:
            return
        n = len(df)
        rows = list(zip(
            [ticker] * n, df.index.asi8.tolist(),
            _col_real(df, 'Stock Splits'),
            _col_dt_ns(df, 'FetchDate'),
            _col_real(df, 'Superseded split'),
            _col_dt_ns(df, 'Superseded split FetchDate'),
        ))
        conn.executemany(
            """INSERT INTO splits
                (ticker, dt_ns, ratio, fetch_date_ns, superseded_split, superseded_fetch_ns)
               VALUES (?,?,?,?,?,?)""",
            rows,
        )

    def _store_earnings_dates(self, conn, ticker: str, df: pd.DataFrame):
        conn.execute("DELETE FROM earnings_dates WHERE ticker=?", (ticker,))
        if df.empty:
            return
        n = len(df)
        rows = list(zip(
            [ticker] * n, df.index.asi8.tolist(),
            _col_real(df, 'Reported EPS'),
            _col_real(df, 'Expected EPS'),
            _col_real(df, 'Surprise(%)'),
            _col_text(df, 'Event Type'),
            _col_dt_ns(df, 'FetchDate'),
            _col_bool(df, 'Date confirmed?'),
        ))
        conn.executemany(
            """INSERT INTO earnings_dates
                (ticker, dt_ns, reported_eps, expected_eps, surprise_pct,
                 event_type, fetch_date_ns, date_confirmed)
               VALUES (?,?,?,?,?,?,?,?)""",
            rows,
        )

    def _store_kv(self, conn, ticker: str, object_name: str, datum):
        blob, json_str = _serialize_kv(datum)
        conn.execute(
            """INSERT OR REPLACE INTO cache_kv (ticker, object_name, data_blob, data_json)
               VALUES (?,?,?,?)""",
            (ticker, object_name, blob, json_str),
        )

    # ------------------------------------------------------------------
    # Internal: structured table readers
    # ------------------------------------------------------------------

    def _make_index(self, ns_list: list, tz_name: str | None) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex(pd.to_datetime(ns_list, unit='ns', utc=True))
        if tz_name:
            idx = idx.tz_convert(tz_name)
        return idx

    def _read_price_history(self, ticker: str, interval: str,
                             tz_name: str | None) -> pd.DataFrame | None:
        rows = self._conn().execute(
            "SELECT * FROM price_history WHERE ticker=? AND interval=? ORDER BY dt_ns",
            (ticker, interval),
        ).fetchall()
        if not rows:
            return None

        index = self._make_index([r['dt_ns'] for r in rows], tz_name)

        df = pd.DataFrame(index=index)
        df.index.name = None

        df['Open']         = pd.array([r['open']      for r in rows], dtype='float64')
        df['High']         = pd.array([r['high']      for r in rows], dtype='float64')
        df['Low']          = pd.array([r['low']       for r in rows], dtype='float64')
        df['Close']        = pd.array([r['close']     for r in rows], dtype='float64')
        df['Volume']       = pd.array([r['volume']    for r in rows], dtype='float64')
        df['Dividends']    = pd.array([r['dividends'] for r in rows], dtype='float64')
        df['Stock Splits'] = pd.array([r['splits']    for r in rows], dtype='float64')

        # FetchDate: UTC-aware datetime column
        df['FetchDate'] = pd.DatetimeIndex(
            pd.to_datetime([r['fetch_date_ns'] if r['fetch_date_ns'] is not None
                            else pd.NaT for r in rows], unit='ns', utc=True)
        )

        df['Final?'] = pd.array(
            [_int_to_bool(r['final']) for r in rows],
            dtype='boolean',
        )

        # Optional columns — only include if at least one non-NULL value
        _opt_bool = [('c_check', 'C-Check?'), ('repaired', 'Repaired?')]
        for sql_col, df_col in _opt_bool:
            vals = [r[sql_col] for r in rows]
            if any(v is not None for v in vals):
                df[df_col] = pd.array(
                    [_int_to_bool(v) for v in vals],
                    dtype='boolean',
                )

        _opt_real = [('csf', 'CSF'), ('cdf', 'CDF')]
        for sql_col, df_col in _opt_real:
            vals = [r[sql_col] for r in rows]
            if any(v is not None for v in vals):
                df[df_col] = pd.array(vals, dtype='float64')

        _opt_dt = [('last_div_adj_ns', 'LastDivAdjustDt'),
                   ('last_split_adj_ns', 'LastSplitAdjustDt')]
        for sql_col, df_col in _opt_dt:
            vals = [r[sql_col] for r in rows]
            if any(v is not None for v in vals):
                df[df_col] = pd.DatetimeIndex(
                    pd.to_datetime([v if v is not None else pd.NaT for v in vals],
                                   unit='ns', utc=True)
                )

        return df

    def _read_dividends(self, ticker: str, tz_name: str | None) -> pd.DataFrame | None:
        rows = self._conn().execute(
            "SELECT * FROM dividends WHERE ticker=? ORDER BY dt_ns", (ticker,)
        ).fetchall()
        if not rows:
            return None

        index = self._make_index([r['dt_ns'] for r in rows], tz_name)

        def _fetch_dates(key):
            return pd.DatetimeIndex(
                pd.to_datetime([r[key] if r[key] is not None else pd.NaT for r in rows],
                               unit='ns', utc=True)
            )

        df = pd.DataFrame({
            'Dividends':               pd.array([r['amount']              for r in rows], dtype='float64'),
            'Back Adj.':               pd.array([r['back_adj']            for r in rows], dtype='float64'),
            'FetchDate':               _fetch_dates('fetch_date_ns'),
            'Close before':            pd.array([r['close_before']        for r in rows], dtype='float64'),
            'Close repaired?':         pd.array([_int_to_bool(r['close_repaired']) for r in rows], dtype='boolean'),
            'Superseded div':          pd.array([r['superseded_div']      for r in rows], dtype='float64'),
            'Superseded back adj.':    pd.array([r['superseded_back_adj'] for r in rows], dtype='float64'),
            'Superseded div FetchDate': _fetch_dates('superseded_fetch_ns'),
        }, index=index)

        return df

    def _read_splits(self, ticker: str, tz_name: str | None) -> pd.DataFrame | None:
        rows = self._conn().execute(
            "SELECT * FROM splits WHERE ticker=? ORDER BY dt_ns", (ticker,)
        ).fetchall()
        if not rows:
            return None

        index = self._make_index([r['dt_ns'] for r in rows], tz_name)

        def _fetch_dates(key):
            return pd.DatetimeIndex(
                pd.to_datetime([r[key] if r[key] is not None else pd.NaT for r in rows],
                               unit='ns', utc=True)
            )

        df = pd.DataFrame({
            'Stock Splits':              pd.array([r['ratio']           for r in rows], dtype='float64'),
            'FetchDate':                 _fetch_dates('fetch_date_ns'),
            'Superseded split':          pd.array([r['superseded_split'] for r in rows], dtype='float64'),
            'Superseded split FetchDate': _fetch_dates('superseded_fetch_ns'),
        }, index=index)

        return df

    def _read_earnings_dates(self, ticker: str, tz_name: str | None) -> pd.DataFrame | None:
        rows = self._conn().execute(
            "SELECT * FROM earnings_dates WHERE ticker=? ORDER BY dt_ns", (ticker,)
        ).fetchall()
        if not rows:
            return None

        index = self._make_index([r['dt_ns'] for r in rows], tz_name)

        def _fetch_dates(key):
            return pd.DatetimeIndex(
                pd.to_datetime([r[key] if r[key] is not None else pd.NaT for r in rows],
                               unit='ns', utc=True)
            )

        df = pd.DataFrame({
            'Reported EPS':  pd.array([r['reported_eps']  for r in rows], dtype='float64'),
            'Expected EPS':  pd.array([r['expected_eps']  for r in rows], dtype='float64'),
            'Surprise(%)':   pd.array([r['surprise_pct']  for r in rows], dtype='float64'),
            'Event Type':    [r['event_type']              for r in rows],
            'FetchDate':     _fetch_dates('fetch_date_ns'),
            'Date confirmed?': pd.array([_int_to_bool(r['date_confirmed']) for r in rows], dtype='boolean'),
        }, index=index)

        return df

    def _read_kv(self, ticker: str, object_name: str):
        row = self._conn().execute(
            "SELECT data_blob, data_json FROM cache_kv WHERE ticker=? AND object_name=?",
            (ticker, object_name),
        ).fetchone()
        if row is None:
            return None
        return _deserialize_kv(row['data_blob'], row['data_json'])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_datum_cached(self, ticker: str, object_name: str) -> bool:
        table, extra = _route(object_name)
        conn = self._conn()
        if table == 'price_history':
            row = conn.execute(
                "SELECT 1 FROM price_history WHERE ticker=? AND interval=? LIMIT 1",
                (ticker, extra),
            ).fetchone()
        elif table in _STRUCTURED:
            row = conn.execute(
                f"SELECT 1 FROM {table} WHERE ticker=? LIMIT 1",
                (ticker,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT 1 FROM cache_kv WHERE ticker=? AND object_name=? LIMIT 1",
                (ticker, object_name),
            ).fetchone()
        return row is not None

    def read_datum(self, ticker: str, object_name: str,
                   return_metadata_too: bool = False):
        table, extra = _route(object_name)

        # Expiry only applies to KV objects
        if table == 'cache_kv' and self._check_and_delete_if_expired(ticker, object_name):
            return (None, None) if return_metadata_too else None

        tz_name, md, expiry_ns = self._read_object_metadata(ticker, object_name)

        if table == 'price_history':
            data = self._read_price_history(ticker, extra, tz_name)
        elif table == 'dividends':
            data = self._read_dividends(ticker, tz_name)
        elif table == 'splits':
            data = self._read_splits(ticker, tz_name)
        elif table == 'earnings_dates':
            data = self._read_earnings_dates(ticker, tz_name)
        else:
            data = self._read_kv(ticker, object_name)

        if data is None:
            return (None, None) if return_metadata_too else None

        if expiry_ns is not None:
            expiry_ts = _ns_to_ts(expiry_ns)
            if md is None:
                md = {'__expiry__': expiry_ts}
            else:
                md['__expiry__'] = expiry_ts

        return (data, md) if return_metadata_too else data

    def store_datum(self, ticker: str, object_name: str, datum,
                    expiry=None, metadata=None):
        if datum is None:
            self.delete_datum(ticker, object_name)
            return

        table, extra = _route(object_name)

        # Derive timezone name from DataFrame index (if applicable)
        tz_name = None
        if isinstance(datum, (pd.DataFrame, pd.Series)):
            tz = getattr(datum.index, 'tz', None)
            if tz is not None:
                tz_name = str(tz)

        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            if table == 'price_history':
                self._store_price_history(conn, ticker, extra, datum)
            elif table == 'dividends':
                self._store_dividends(conn, ticker, datum)
            elif table == 'splits':
                self._store_splits(conn, ticker, datum)
            elif table == 'earnings_dates':
                self._store_earnings_dates(conn, ticker, datum)
            else:
                self._store_kv(conn, ticker, object_name, datum)

            self._upsert_object_metadata(conn, ticker, object_name,
                                          metadata, expiry, tz_name)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def delete_datum(self, ticker: str, object_name: str):
        table, extra = _route(object_name)
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            if table == 'price_history':
                conn.execute(
                    "DELETE FROM price_history WHERE ticker=? AND interval=?",
                    (ticker, extra),
                )
            elif table in _STRUCTURED:
                conn.execute(f"DELETE FROM {table} WHERE ticker=?", (ticker,))
            else:
                conn.execute(
                    "DELETE FROM cache_kv WHERE ticker=? AND object_name=?",
                    (ticker, object_name),
                )
            conn.execute(
                "DELETE FROM object_metadata WHERE ticker=? AND object_name=?",
                (ticker, object_name),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def read_metadata_key(self, ticker: str, object_name: str, key: str):
        row = self._conn().execute(
            "SELECT metadata_json FROM object_metadata WHERE ticker=? AND object_name=?",
            (ticker, object_name),
        ).fetchone()
        if row is None:
            return None
        md = _deserialize_metadata(row['metadata_json'])
        return None if md is None else md.get(key)

    def write_metadata_key(self, ticker: str, object_name: str, key: str, value):
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT metadata_json FROM object_metadata WHERE ticker=? AND object_name=?",
                (ticker, object_name),
            ).fetchone()
            if row is None:
                md = {key: value} if value is not None else {}
                conn.execute(
                    """INSERT INTO object_metadata
                        (ticker, object_name, metadata_json, expiry_ns, tz_name)
                       VALUES (?,?,?,NULL,NULL)""",
                    (ticker, object_name, _serialize_metadata(md)),
                )
            else:
                md = _deserialize_metadata(row['metadata_json']) or {}
                if value is None:
                    md.pop(key, None)
                else:
                    md[key] = value
                conn.execute(
                    "UPDATE object_metadata SET metadata_json=? WHERE ticker=? AND object_name=?",
                    (_serialize_metadata(md) or None, ticker, object_name),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def list_tickers(self) -> list:
        rows = self._conn().execute("""
            SELECT DISTINCT ticker FROM cache_kv
            UNION
            SELECT DISTINCT ticker FROM price_history
            UNION
            SELECT DISTINCT ticker FROM dividends
            UNION
            SELECT DISTINCT ticker FROM splits
            UNION
            SELECT DISTINCT ticker FROM earnings_dates
        """).fetchall()
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
                (f"_YFC_/{flag_name}",
                 datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def list_upgrade_flags(self) -> list:
        # Use ESCAPE to treat the '_' characters in '_YFC_/' as literals.
        # Without escaping, '_' is a single-character wildcard in SQL LIKE and
        # the pattern would incorrectly match keys like 'XYFCX/...'.
        rows = self._conn().execute(
            r"SELECT key FROM cache_meta WHERE key LIKE '\_YFC\_/%' ESCAPE '\'"
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
