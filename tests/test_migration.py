"""
Tests for yfc_migrate.py — files ↔ SQLite roundtrip migration.
"""

import os
import pickle
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

from .context import yfc_cache_manager as yfcm
import yfinance_cache as yfc
from yfinance_cache.yfc_sqlite_manager import SqliteCacheBackend


class Test_Migration(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        yfcm.SetCacheDirpath(self.tempdir.name)
        # Ensure file backend
        yfcm._option_manager.cache.cache_backend = 'files'
        # Reset SQLite singleton so a new one is created for each test
        yfcm._sqlite_backend = None

    def tearDown(self):
        # Restore file backend before cleanup
        yfcm._option_manager.cache.cache_backend = 'files'
        yfcm._sqlite_backend = None
        self.tempdir.cleanup()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _switch_to_sqlite(self):
        yfcm._option_manager.cache.cache_backend = 'sqlite'
        yfcm._sqlite_backend = None  # force fresh backend on new db_path

    def _switch_to_files(self):
        if yfcm._sqlite_backend is not None:
            yfcm._sqlite_backend.close()
            yfcm._sqlite_backend = None
        yfcm._option_manager.cache.cache_backend = 'files'

    # ------------------------------------------------------------------
    # Files → SQLite
    # ------------------------------------------------------------------

    def test_migrate_to_sqlite_basic(self):
        # Populate file cache
        yfcm.StoreCacheDatum('AAPL', 'info', {'name': 'Apple'})
        yfcm.StoreCacheDatum('AAPL', 'isin', 'US0378331005')
        yfcm.StoreCachePackedDatum('AAPL', 'balance_sheet', 1.0)
        yfcm.StoreCachePackedDatum('AAPL', 'cashflow', 2.0)

        result = yfc.migrate_to_sqlite(cache_dir=self.tempdir.name, progress=False)
        self.assertEqual(result['errors'], [])
        self.assertGreaterEqual(result['objects'], 4)

        # Verify via SQLite backend directly
        db_path = os.path.join(self.tempdir.name, 'yfc_cache.db')
        backend = SqliteCacheBackend(db_path)
        self.assertEqual(backend.read_datum('AAPL', 'info'), {'name': 'Apple'})
        self.assertEqual(backend.read_datum('AAPL', 'isin'), 'US0378331005')
        self.assertEqual(backend.read_datum('AAPL', 'balance_sheet'), 1.0)
        self.assertEqual(backend.read_datum('AAPL', 'cashflow'), 2.0)
        backend.close()

    def test_migrate_to_sqlite_dataframe(self):
        df = pd.DataFrame({'Open': [100.0, 101.0], 'Close': [102.0, 103.0]})
        yfcm.StoreCacheDatum('MSFT', 'history-1d', df)

        yfc.migrate_to_sqlite(cache_dir=self.tempdir.name, progress=False)

        db_path = os.path.join(self.tempdir.name, 'yfc_cache.db')
        backend = SqliteCacheBackend(db_path)
        result = backend.read_datum('MSFT', 'history-1d')
        pd.testing.assert_frame_equal(result, df)
        backend.close()

    def test_migrate_to_sqlite_preserves_metadata(self):
        from datetime import date as _date
        # Use non-date-string values to avoid the JSON codec date decode
        md = {'source': 'yahoo', 'version': 2}
        yfcm.StoreCacheDatum('AAPL', 'info', {'x': 1}, metadata=md)

        yfc.migrate_to_sqlite(cache_dir=self.tempdir.name, progress=False)

        db_path = os.path.join(self.tempdir.name, 'yfc_cache.db')
        backend = SqliteCacheBackend(db_path)
        _, result_md = backend.read_datum('AAPL', 'info', return_metadata_too=True)
        backend.close()
        self.assertEqual(result_md.get('source'), 'yahoo')
        self.assertEqual(result_md.get('version'), 2)

    def test_migrate_to_sqlite_preserves_expiry(self):
        exp = datetime.now(timezone.utc) + timedelta(hours=2)
        yfcm.StoreCacheDatum('AAPL', 'info', 99, expiry=exp)

        yfc.migrate_to_sqlite(cache_dir=self.tempdir.name, progress=False)

        db_path = os.path.join(self.tempdir.name, 'yfc_cache.db')
        backend = SqliteCacheBackend(db_path)
        _, md = backend.read_datum('AAPL', 'info', return_metadata_too=True)
        backend.close()
        self.assertIsNotNone(md)
        self.assertIn('__expiry__', md)
        self.assertAlmostEqual(md['__expiry__'].timestamp(), exp.timestamp(), delta=2)

    def test_migrate_to_sqlite_copies_upgrade_flags(self):
        # Create a sentinel file
        yfc_dp = os.path.join(self.tempdir.name, '_YFC_')
        os.makedirs(yfc_dp, exist_ok=True)
        open(os.path.join(yfc_dp, 'have-done-something'), 'w').close()

        yfc.migrate_to_sqlite(cache_dir=self.tempdir.name, progress=False)

        db_path = os.path.join(self.tempdir.name, 'yfc_cache.db')
        backend = SqliteCacheBackend(db_path)
        self.assertTrue(backend.has_upgrade_flag('have-done-something'))
        backend.close()

    def test_migrate_to_sqlite_multiple_tickers(self):
        for ticker in ['AAPL', 'MSFT', 'GOOG']:
            yfcm.StoreCacheDatum(ticker, 'info', {'ticker': ticker})

        result = yfc.migrate_to_sqlite(cache_dir=self.tempdir.name, progress=False)
        self.assertEqual(result['tickers'], 3)

    def test_migrate_to_sqlite_dry_run(self):
        yfcm.StoreCacheDatum('AAPL', 'info', {'x': 1})
        yfc.migrate_to_sqlite(cache_dir=self.tempdir.name, progress=False, dry_run=True)
        db_path = os.path.join(self.tempdir.name, 'yfc_cache.db')
        # DB may be created but should have no rows
        if os.path.isfile(db_path):
            backend = SqliteCacheBackend(db_path)
            self.assertFalse(backend.is_datum_cached('AAPL', 'info'))
            backend.close()

    # ------------------------------------------------------------------
    # SQLite → Files
    # ------------------------------------------------------------------

    def test_migrate_to_files_basic(self):
        # Populate SQLite
        self._switch_to_sqlite()
        yfcm.StoreCacheDatum('AAPL', 'info', {'name': 'Apple'})
        yfcm.StoreCacheDatum('AAPL', 'isin', 'US0378331005')
        yfcm.StoreCachePackedDatum('AAPL', 'balance_sheet', 5.0)
        self._switch_to_files()

        result = yfc.migrate_to_files(cache_dir=self.tempdir.name, progress=False)
        self.assertEqual(result['errors'], [], result['errors'])
        self.assertGreaterEqual(result['objects'], 3)

        # Verify via file backend
        self.assertEqual(yfcm.ReadCacheDatum('AAPL', 'info'), {'name': 'Apple'})
        self.assertEqual(yfcm.ReadCacheDatum('AAPL', 'isin'), 'US0378331005')
        self.assertEqual(yfcm.ReadCachePackedDatum('AAPL', 'balance_sheet'), 5.0)

    def test_migrate_to_files_reconstructs_pack_file(self):
        self._switch_to_sqlite()
        yfcm.StoreCachePackedDatum('AAPL', 'balance_sheet', 10.0)
        yfcm.StoreCachePackedDatum('AAPL', 'cashflow', 20.0)
        yfcm.StoreCachePackedDatum('AAPL', 'income_stmt', 30.0)
        self._switch_to_files()

        yfc.migrate_to_files(cache_dir=self.tempdir.name, progress=False)

        # The pack file should exist on disk
        pack_fp = os.path.join(self.tempdir.name, 'AAPL', 'annuals.pkl')
        self.assertTrue(os.path.isfile(pack_fp))

        with open(pack_fp, 'rb') as fh:
            packed = pickle.load(fh)
        self.assertIn('balance_sheet', packed)
        self.assertIn('cashflow', packed)
        self.assertIn('income_stmt', packed)
        self.assertEqual(packed['balance_sheet']['data'], 10.0)

    def test_migrate_to_files_dry_run(self):
        self._switch_to_sqlite()
        yfcm.StoreCacheDatum('AAPL', 'info', {'x': 1})
        self._switch_to_files()

        yfc.migrate_to_files(cache_dir=self.tempdir.name, progress=False, dry_run=True)

        # No files should be written for AAPL/info
        fp_json = os.path.join(self.tempdir.name, 'AAPL', 'info.json')
        fp_pkl  = os.path.join(self.tempdir.name, 'AAPL', 'info.pkl')
        self.assertFalse(os.path.isfile(fp_json))
        self.assertFalse(os.path.isfile(fp_pkl))

    def test_migrate_to_files_missing_db(self):
        result = yfc.migrate_to_files(cache_dir=self.tempdir.name, progress=False)
        self.assertEqual(result['tickers'], 0)
        self.assertTrue(len(result['errors']) > 0)

    # ------------------------------------------------------------------
    # Full roundtrip (files → SQLite → files)
    # ------------------------------------------------------------------

    def test_full_roundtrip(self):
        # Populate file cache with a variety of types
        yfcm.StoreCacheDatum('AAPL', 'isin', 'US0378331005')
        yfcm.StoreCacheDatum('AAPL', 'info', {'exchange': 'NMS', 'currency': 'USD'})
        df = pd.DataFrame({'Close': [150.0, 151.0, 152.0]})
        yfcm.StoreCacheDatum('AAPL', 'history-1d', df)
        yfcm.StoreCachePackedDatum('AAPL', 'balance_sheet', 999.0)
        yfcm.StoreCachePackedDatum('AAPL', 'cashflow', 888.0)

        # → SQLite
        r1 = yfc.migrate_to_sqlite(cache_dir=self.tempdir.name, progress=False)
        self.assertEqual(r1['errors'], [])

        # Remove the original files so we can verify restoration
        import shutil
        aapl_dir = os.path.join(self.tempdir.name, 'AAPL')
        shutil.rmtree(aapl_dir)

        # Switch to SQLite and verify data is there
        self._switch_to_sqlite()
        self.assertEqual(yfcm.ReadCacheDatum('AAPL', 'isin'), 'US0378331005')
        self._switch_to_files()

        # → Files
        r2 = yfc.migrate_to_files(cache_dir=self.tempdir.name, progress=False)
        self.assertEqual(r2['errors'], [], r2['errors'])

        # Verify via file backend
        self.assertEqual(yfcm.ReadCacheDatum('AAPL', 'isin'), 'US0378331005')
        self.assertEqual(yfcm.ReadCacheDatum('AAPL', 'info'), {'exchange': 'NMS', 'currency': 'USD'})
        result_df = yfcm.ReadCacheDatum('AAPL', 'history-1d')
        pd.testing.assert_frame_equal(result_df, df)
        self.assertEqual(yfcm.ReadCachePackedDatum('AAPL', 'balance_sheet'), 999.0)
        self.assertEqual(yfcm.ReadCachePackedDatum('AAPL', 'cashflow'), 888.0)

    # ------------------------------------------------------------------
    # Error tolerance
    # ------------------------------------------------------------------

    def test_corrupt_file_skipped(self):
        # Write a valid entry and a corrupt .pkl file
        yfcm.StoreCacheDatum('AAPL', 'info', {'x': 1})
        corrupt_fp = os.path.join(self.tempdir.name, 'AAPL', 'corrupt.pkl')
        with open(corrupt_fp, 'wb') as fh:
            fh.write(b'not valid pickle data')

        result = yfc.migrate_to_sqlite(cache_dir=self.tempdir.name, progress=False)
        # Migration should still report the good object
        self.assertGreaterEqual(result['objects'], 1)
        # Corrupt file should be listed in errors
        self.assertTrue(any('corrupt' in e for e in result['errors']), result['errors'])


if __name__ == '__main__':
    unittest.main()
