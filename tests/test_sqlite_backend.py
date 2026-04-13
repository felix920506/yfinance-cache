"""
Tests specific to the SqliteCacheBackend class.
General cache-semantics tests are in test_cache.py (parametrised for both backends).
"""

import os
import sqlite3
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd
from zoneinfo import ZoneInfo

from .context import yfc_cache_manager as yfcm
from yfinance_cache.yfc_sqlite_manager import SqliteCacheBackend


class Test_SqliteBackend(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tempdir.name, 'test.db')
        self.backend = SqliteCacheBackend(self.db_path)

    def tearDown(self):
        self.backend.close()
        self.tempdir.cleanup()

    # ------------------------------------------------------------------
    # Schema / connection
    # ------------------------------------------------------------------

    def test_wal_mode(self):
        row = self.backend._conn().execute("PRAGMA journal_mode").fetchone()
        self.assertEqual(row[0], 'wal')

    def test_schema_idempotent(self):
        """Creating a second backend on the same DB must not raise."""
        backend2 = SqliteCacheBackend(self.db_path)
        backend2.close()

    def test_db_file_created(self):
        self.assertTrue(os.path.isfile(self.db_path))

    # ------------------------------------------------------------------
    # Basic CRUD
    # ------------------------------------------------------------------

    def test_store_and_read_int(self):
        self.backend.store_datum('AAPL', 'score', 42)
        result = self.backend.read_datum('AAPL', 'score')
        self.assertEqual(result, 42)

    def test_store_and_read_dict(self):
        d = {'a': 1, 'b': 'hello'}
        self.backend.store_datum('AAPL', 'info', d)
        result = self.backend.read_datum('AAPL', 'info')
        self.assertEqual(result, d)

    def test_store_and_read_dataframe(self):
        # Use a non-structured key so the DataFrame goes through the KV path
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4.0, 5.0, 6.0]})
        self.backend.store_datum('AAPL', 'some-df', df)
        result = self.backend.read_datum('AAPL', 'some-df')
        pd.testing.assert_frame_equal(result, df)

    def test_is_datum_cached_true(self):
        self.backend.store_datum('AAPL', 'info', {'x': 1})
        self.assertTrue(self.backend.is_datum_cached('AAPL', 'info'))

    def test_is_datum_cached_false(self):
        self.assertFalse(self.backend.is_datum_cached('AAPL', 'missing'))

    def test_delete_datum(self):
        self.backend.store_datum('AAPL', 'info', 1)
        self.backend.delete_datum('AAPL', 'info')
        self.assertFalse(self.backend.is_datum_cached('AAPL', 'info'))

    def test_store_none_deletes(self):
        self.backend.store_datum('AAPL', 'info', 1)
        self.backend.store_datum('AAPL', 'info', None)
        self.assertFalse(self.backend.is_datum_cached('AAPL', 'info'))

    def test_overwrite_preserves_metadata_when_none_passed(self):
        self.backend.store_datum('AAPL', 'info', 1, metadata={'k': 'v'})
        self.backend.store_datum('AAPL', 'info', 2)  # metadata=None → preserve
        _, md = self.backend.read_datum('AAPL', 'info', return_metadata_too=True)
        self.assertEqual(md, {'k': 'v'})

    def test_overwrite_replaces_metadata_when_given(self):
        self.backend.store_datum('AAPL', 'info', 1, metadata={'k': 'v'})
        self.backend.store_datum('AAPL', 'info', 2, metadata={'k2': 'v2'})
        _, md = self.backend.read_datum('AAPL', 'info', return_metadata_too=True)
        self.assertEqual(md, {'k2': 'v2'})

    # ------------------------------------------------------------------
    # Expiry
    # ------------------------------------------------------------------

    def test_expiry_future_not_deleted(self):
        exp = datetime.now(timezone.utc) + timedelta(hours=1)
        self.backend.store_datum('AAPL', 'info', 99, expiry=exp)
        result = self.backend.read_datum('AAPL', 'info')
        self.assertEqual(result, 99)

    def test_expiry_past_returns_none_and_deletes_row(self):
        exp = datetime.now(timezone.utc) - timedelta(seconds=1)
        self.backend.store_datum('AAPL', 'info', 99, expiry=exp)
        result = self.backend.read_datum('AAPL', 'info')
        self.assertIsNone(result)
        self.assertFalse(self.backend.is_datum_cached('AAPL', 'info'))

    def test_expiry_injected_into_metadata(self):
        exp = datetime.now(timezone.utc) + timedelta(hours=1)
        self.backend.store_datum('AAPL', 'info', 99, expiry=exp)
        _, md = self.backend.read_datum('AAPL', 'info', return_metadata_too=True)
        self.assertIn('__expiry__', md)
        self.assertAlmostEqual(
            md['__expiry__'].timestamp(), exp.timestamp(), delta=1
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def test_read_metadata_key(self):
        from datetime import date as _date
        # JsonDecodeDict decodes ISO date strings to datetime.date objects
        self.backend.store_datum('AAPL', 'info', 1, metadata={'fetchDate': '2024-01-01'})
        v = self.backend.read_metadata_key('AAPL', 'info', 'fetchDate')
        self.assertEqual(v, _date(2024, 1, 1))

    def test_read_metadata_key_missing(self):
        self.backend.store_datum('AAPL', 'info', 1)
        v = self.backend.read_metadata_key('AAPL', 'info', 'nokey')
        self.assertIsNone(v)

    def test_write_metadata_key_updates(self):
        self.backend.store_datum('AAPL', 'info', 1, metadata={'k': 'old'})
        self.backend.write_metadata_key('AAPL', 'info', 'k', 'new')
        v = self.backend.read_metadata_key('AAPL', 'info', 'k')
        self.assertEqual(v, 'new')

    def test_write_metadata_key_delete_with_none(self):
        self.backend.store_datum('AAPL', 'info', 1, metadata={'k': 'v', 'k2': 'v2'})
        self.backend.write_metadata_key('AAPL', 'info', 'k', None)
        v = self.backend.read_metadata_key('AAPL', 'info', 'k')
        self.assertIsNone(v)
        # Other key still present
        v2 = self.backend.read_metadata_key('AAPL', 'info', 'k2')
        self.assertEqual(v2, 'v2')

    # ------------------------------------------------------------------
    # Pack name preservation
    # ------------------------------------------------------------------

    def test_packed_objects_readable(self):
        self.backend.store_datum('AAPL', 'balance_sheet', 100.0)
        self.backend.store_datum('AAPL', 'cashflow', 200.0)
        self.assertEqual(self.backend.read_datum('AAPL', 'balance_sheet'), 100.0)
        self.assertEqual(self.backend.read_datum('AAPL', 'cashflow'), 200.0)

    # ------------------------------------------------------------------
    # Ticker listing
    # ------------------------------------------------------------------

    def test_list_tickers(self):
        self.backend.store_datum('AAPL', 'info', 1)
        self.backend.store_datum('MSFT', 'info', 2)
        self.backend.store_datum('GOOG', 'info', 3)
        tickers = set(self.backend.list_tickers())
        self.assertEqual(tickers, {'AAPL', 'MSFT', 'GOOG'})

    # ------------------------------------------------------------------
    # Upgrade flags
    # ------------------------------------------------------------------

    def test_upgrade_flag_round_trip(self):
        self.assertFalse(self.backend.has_upgrade_flag('have-done-thing'))
        self.backend.set_upgrade_flag('have-done-thing')
        self.assertTrue(self.backend.has_upgrade_flag('have-done-thing'))

    def test_list_upgrade_flags(self):
        self.backend.set_upgrade_flag('flag-a')
        self.backend.set_upgrade_flag('flag-b')
        flags = set(self.backend.list_upgrade_flags())
        self.assertIn('flag-a', flags)
        self.assertIn('flag-b', flags)

    def test_delete_upgrade_flag(self):
        self.backend.set_upgrade_flag('flag-x')
        self.backend.delete_upgrade_flag('flag-x')
        self.assertFalse(self.backend.has_upgrade_flag('flag-x'))

    def test_list_upgrade_flags_no_wildcard_false_positive(self):
        """LIKE '_YFC_/%' must not match keys whose '_' chars are SQL wildcards.

        Without ESCAPE the pattern '_YFC_/%' would match e.g. 'XYFC_/foo'
        because '_' is a single-char wildcard in SQL LIKE.
        """
        conn = self.backend._conn()
        # Insert a key that LIKE '_YFC_/%' without ESCAPE would match
        conn.execute("INSERT OR REPLACE INTO cache_meta (key, value) VALUES (?, ?)",
                     ('XYFCX/impostor', 'bad'))
        conn.commit()
        flags = self.backend.list_upgrade_flags()
        self.assertNotIn('impostor', flags,
                         "list_upgrade_flags() matched a key via SQL wildcard '_'")

    # ------------------------------------------------------------------
    # Concurrency
    # ------------------------------------------------------------------

    def test_concurrent_writes(self):
        errors = []
        writes_per_thread = 50
        num_threads = 10

        def worker(thread_id):
            for i in range(writes_per_thread):
                try:
                    self.backend.store_datum(
                        f'TICKER{thread_id}', f'obj{i}', thread_id * 1000 + i
                    )
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], errors)
        # Verify counts
        conn = self.backend._conn()
        count = conn.execute("SELECT COUNT(*) FROM cache_kv").fetchone()[0]
        self.assertEqual(count, num_threads * writes_per_thread)

    def test_concurrent_expiry_read(self):
        """Two threads reading the same expired key must not raise."""
        exp = datetime.now(timezone.utc) - timedelta(seconds=1)
        self.backend.store_datum('AAPL', 'info', 99, expiry=exp)
        errors = []

        def reader():
            try:
                self.backend.read_datum('AAPL', 'info')
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], errors)

    # ------------------------------------------------------------------
    # Structured tables
    # ------------------------------------------------------------------

    def _make_price_df(self, tz='US/Eastern', with_optional=False):
        """Minimal price DataFrame that matches the structured schema."""
        tz_info = ZoneInfo(tz)
        index = pd.DatetimeIndex([
            datetime(2024, 1, 2, 9, 30, tzinfo=tz_info),
            datetime(2024, 1, 3, 9, 30, tzinfo=tz_info),
            datetime(2024, 1, 4, 9, 30, tzinfo=tz_info),
        ])
        df = pd.DataFrame({
            'Open':         [150.0, 151.0, 152.0],
            'High':         [155.0, 156.0, 157.0],
            'Low':          [149.0, 150.0, 151.0],
            'Close':        [153.0, 154.0, 155.0],
            'Volume':       [1e6, 1.1e6, 1.2e6],
            'Dividends':    [0.0, 0.0, 0.0],
            'Stock Splits': [0.0, 0.0, 0.0],
            'FetchDate':    pd.DatetimeIndex([
                pd.Timestamp('2024-01-03', tz='UTC'),
                pd.Timestamp('2024-01-04', tz='UTC'),
                pd.Timestamp('2024-01-05', tz='UTC'),
            ]),
            'Final?': [True, True, False],
        }, index=index)
        if with_optional:
            df['CSF']       = [1.0, 1.0, 1.0]
            df['CDF']       = [1.0, 1.0, 1.0]
            df['C-Check?']  = [True, True, False]
            df['Repaired?'] = [False, False, False]
        return df

    def test_price_history_structured(self):
        """Store a price DataFrame, verify SQL rows, read back and compare."""
        df = self._make_price_df()
        self.backend.store_datum('AAPL', 'history-1d', df)

        # Check SQL rows exist in price_history
        conn = self.backend._conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM price_history WHERE ticker='AAPL' AND interval='1d'"
        ).fetchone()[0]
        self.assertEqual(count, 3)

        # cache_kv should NOT have this entry
        kv_count = conn.execute(
            "SELECT COUNT(*) FROM cache_kv WHERE ticker='AAPL'"
        ).fetchone()[0]
        self.assertEqual(kv_count, 0)

        result = self.backend.read_datum('AAPL', 'history-1d')
        self.assertIsNotNone(result)
        # Compare core columns. Boolean columns are read back as pandas nullable
        # boolean (dtype='boolean') to preserve NULL semantics, so skip dtype check.
        for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends',
                    'Stock Splits', 'Final?']:
            pd.testing.assert_series_equal(
                result[col].reset_index(drop=True),
                df[col].reset_index(drop=True),
                check_names=False,
                check_dtype=False,
            )

    def test_price_history_timezone(self):
        """Index timezone is preserved round-trip."""
        df = self._make_price_df(tz='US/Eastern')
        self.backend.store_datum('AAPL', 'history-1d', df)
        result = self.backend.read_datum('AAPL', 'history-1d')
        self.assertEqual(str(result.index.tz), 'US/Eastern')
        # Compare UTC nanosecond values; both represent the same instants
        import numpy as np
        np.testing.assert_array_equal(df.index.asi8, result.index.asi8)

    def test_price_history_optional_columns_absent(self):
        """Optional columns are NOT present when data had none."""
        df = self._make_price_df(with_optional=False)
        self.backend.store_datum('AAPL', 'history-1d', df)
        result = self.backend.read_datum('AAPL', 'history-1d')
        for col in ('CSF', 'CDF', 'C-Check?', 'Repaired?',
                    'LastDivAdjustDt', 'LastSplitAdjustDt'):
            self.assertNotIn(col, result.columns, f"Column {col!r} should be absent")

    def test_price_history_optional_columns_present(self):
        """Optional columns appear when data has at least one non-NULL value."""
        df = self._make_price_df(with_optional=True)
        self.backend.store_datum('AAPL', 'history-1d', df)
        result = self.backend.read_datum('AAPL', 'history-1d')
        for col in ('CSF', 'CDF', 'C-Check?', 'Repaired?'):
            self.assertIn(col, result.columns, f"Column {col!r} should be present")

    def test_boolean_null_preserved_as_pd_na(self):
        """NULL booleans must round-trip as pd.NA, not False."""
        tz_info = ZoneInfo('US/Eastern')
        index = pd.DatetimeIndex([datetime(2024, 1, 2, 9, 30, tzinfo=tz_info)])
        # Use nullable boolean so pd.NA can be stored
        df = pd.DataFrame({
            'Open': [150.0], 'High': [155.0], 'Low': [149.0], 'Close': [153.0],
            'Volume': [1e6], 'Dividends': [0.0], 'Stock Splits': [0.0],
            'FetchDate': pd.DatetimeIndex([pd.Timestamp('2024-01-03', tz='UTC')]),
            'Final?': pd.array([pd.NA], dtype='boolean'),
        }, index=index)
        self.backend.store_datum('AAPL', 'history-1d', df)
        result = self.backend.read_datum('AAPL', 'history-1d')
        # Must come back as pd.NA, not False
        self.assertTrue(pd.isna(result['Final?'].iloc[0]),
                        "NULL Final? should round-trip as pd.NA, not False")

    def test_dividends_structured(self):
        """Store and read a dividends DataFrame via the structured table."""
        tz_info = ZoneInfo('US/Eastern')
        index = pd.DatetimeIndex([
            datetime(2024, 2, 15, 9, 30, tzinfo=tz_info),
            datetime(2024, 5, 16, 9, 30, tzinfo=tz_info),
        ])
        df = pd.DataFrame({
            'Dividends':               [0.24, 0.24],
            'Back Adj.':               [0.24, 0.24],
            'FetchDate':               pd.DatetimeIndex([
                pd.Timestamp('2024-02-16', tz='UTC'),
                pd.Timestamp('2024-05-17', tz='UTC'),
            ]),
            'Close before':            [185.0, 190.0],
            'Close repaired?':         [False, False],
            'Superseded div':          [float('nan'), float('nan')],
            'Superseded back adj.':    [float('nan'), float('nan')],
            'Superseded div FetchDate': pd.DatetimeIndex(
                [pd.NaT, pd.NaT], dtype='datetime64[ns, UTC]'
            ),
        }, index=index)

        self.backend.store_datum('AAPL', 'dividends', df)

        conn = self.backend._conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM dividends WHERE ticker='AAPL'"
        ).fetchone()[0]
        self.assertEqual(count, 2)

        result = self.backend.read_datum('AAPL', 'dividends')
        self.assertIsNotNone(result)
        pd.testing.assert_series_equal(
            result['Dividends'].reset_index(drop=True),
            df['Dividends'].reset_index(drop=True),
            check_names=False,
        )
        self.assertEqual(str(result.index.tz), 'US/Eastern')
        import numpy as np
        np.testing.assert_array_equal(df.index.asi8, result.index.asi8)

    def test_splits_structured(self):
        """Store and read a splits DataFrame via the structured table."""
        tz_info = ZoneInfo('US/Eastern')
        index = pd.DatetimeIndex([
            datetime(2020, 8, 31, 9, 30, tzinfo=tz_info),
        ])
        df = pd.DataFrame({
            'Stock Splits':              [4.0],
            'FetchDate':                 pd.DatetimeIndex([
                pd.Timestamp('2020-09-01', tz='UTC'),
            ]),
            'Superseded split':          [float('nan')],
            'Superseded split FetchDate': pd.DatetimeIndex(
                [pd.NaT], dtype='datetime64[ns, UTC]'
            ),
        }, index=index)

        self.backend.store_datum('AAPL', 'splits', df)

        conn = self.backend._conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM splits WHERE ticker='AAPL'"
        ).fetchone()[0]
        self.assertEqual(count, 1)

        result = self.backend.read_datum('AAPL', 'splits')
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['Stock Splits'].iloc[0], 4.0)

    def test_earnings_dates_structured(self):
        """Store and read an earnings_dates DataFrame via the structured table."""
        tz_info = ZoneInfo('US/Eastern')
        index = pd.DatetimeIndex([
            datetime(2024, 2, 1, 16, 30, tzinfo=tz_info),
            datetime(2024, 5, 2, 16, 30, tzinfo=tz_info),
        ])
        df = pd.DataFrame({
            'Reported EPS':  [2.18, 1.53],
            'Expected EPS':  [2.10, 1.50],
            'Surprise(%)':   [3.8, 2.0],
            'Event Type':    ['Quarterly', 'Quarterly'],
            'FetchDate':     pd.DatetimeIndex([
                pd.Timestamp('2024-02-02', tz='UTC'),
                pd.Timestamp('2024-05-03', tz='UTC'),
            ]),
            'Date confirmed?': [True, True],
        }, index=index)

        self.backend.store_datum('AAPL', 'earnings_dates', df)

        conn = self.backend._conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM earnings_dates WHERE ticker='AAPL'"
        ).fetchone()[0]
        self.assertEqual(count, 2)

        result = self.backend.read_datum('AAPL', 'earnings_dates')
        self.assertIsNotNone(result)
        self.assertEqual(str(result.index.tz), 'US/Eastern')
        import numpy as np
        np.testing.assert_array_equal(df.index.asi8, result.index.asi8)
        self.assertEqual(list(result['Event Type']), ['Quarterly', 'Quarterly'])
        self.assertTrue(all(result['Date confirmed?']))

    def test_kv_still_works_for_non_structured(self):
        """Non-structured keys (info, calendar, etc.) still use cache_kv."""
        payload = {'name': 'Apple Inc.', 'sector': 'Technology'}
        self.backend.store_datum('AAPL', 'info', payload)

        conn = self.backend._conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM cache_kv WHERE ticker='AAPL' AND object_name='info'"
        ).fetchone()[0]
        self.assertEqual(count, 1)

        result = self.backend.read_datum('AAPL', 'info')
        self.assertEqual(result, payload)

    def test_price_history_replace(self):
        """Storing a new DataFrame for the same (ticker, interval) replaces old rows."""
        df1 = self._make_price_df()
        self.backend.store_datum('AAPL', 'history-1d', df1)

        tz_info = ZoneInfo('US/Eastern')
        index2 = pd.DatetimeIndex([datetime(2024, 1, 5, 9, 30, tzinfo=tz_info)])
        df2 = pd.DataFrame({
            'Open': [160.0], 'High': [165.0], 'Low': [159.0], 'Close': [163.0],
            'Volume': [2e6], 'Dividends': [0.0], 'Stock Splits': [0.0],
            'FetchDate': pd.DatetimeIndex([pd.Timestamp('2024-01-06', tz='UTC')]),
            'Final?': [True],
        }, index=index2)
        self.backend.store_datum('AAPL', 'history-1d', df2)

        conn = self.backend._conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM price_history WHERE ticker='AAPL' AND interval='1d'"
        ).fetchone()[0]
        self.assertEqual(count, 1)

        result = self.backend.read_datum('AAPL', 'history-1d')
        self.assertAlmostEqual(result['Open'].iloc[0], 160.0)

    def test_list_tickers_includes_structured(self):
        """list_tickers() returns tickers from structured tables too."""
        df = self._make_price_df()
        self.backend.store_datum('AAPL', 'history-1d', df)
        self.backend.store_datum('MSFT', 'info', {'x': 1})
        tickers = set(self.backend.list_tickers())
        self.assertIn('AAPL', tickers)
        self.assertIn('MSFT', tickers)


if __name__ == '__main__':
    unittest.main()
