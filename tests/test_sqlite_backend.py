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
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4.0, 5.0, 6.0]})
        self.backend.store_datum('AAPL', 'history-1d', df)
        result = self.backend.read_datum('AAPL', 'history-1d')
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
        count = conn.execute("SELECT COUNT(*) FROM cache_data").fetchone()[0]
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


if __name__ == '__main__':
    unittest.main()
