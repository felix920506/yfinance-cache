"""
Migration utilities for switching between the file and SQLite cache backends.

migrate_to_sqlite() — copy all file-based cache data into yfc_cache.db
migrate_to_files()  — copy all SQLite rows back out as files

Neither function changes yfc.options.cache.cache_backend automatically;
you must do that yourself after confirming the migration succeeded.
"""

import os
import pickle

from . import yfc_cache_manager as yfcm
from . import yfc_dat as yfcd


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _iter_ticker_dirs(cache_dir: str):
    """Yield ticker names found in the file-based cache directory."""
    skip = {'options.json', '_YFC_', 'yfc_cache.db'}
    for name in os.listdir(cache_dir):
        if name in skip:
            continue
        if os.path.isdir(os.path.join(cache_dir, name)):
            yield name


def _load_pack_names() -> dict:
    """Return pack_name -> [object_names] from yfc_cache_manager."""
    return yfcm.packed_data_cats


def _object_name_from_filename(filename: str) -> str | None:
    """Strip .json or .pkl extension, return object name or None."""
    for ext in ('.json', '.pkl'):
        if filename.endswith(ext):
            return filename[:-len(ext)]
    return None


# -------------------------------------------------------------------------
# migrate_to_sqlite
# -------------------------------------------------------------------------

def migrate_to_sqlite(
    cache_dir: str | None = None,
    progress: bool = True,
    dry_run: bool = False,
) -> dict:
    """
    Read all data from the file-based cache and insert it into SQLite.

    Parameters
    ----------
    cache_dir : str, optional
        Path to the cache directory. Defaults to the current cache dir.
    progress : bool
        Print per-ticker progress lines.
    dry_run : bool
        If True, read files but do not write to SQLite.

    Returns
    -------
    dict with keys 'tickers', 'objects', 'errors'.
    """
    from .yfc_sqlite_manager import SqliteCacheBackend

    if cache_dir is None:
        cache_dir = yfcm.GetCacheDirpath()

    db_path = os.path.join(cache_dir, 'yfc_cache.db')
    backend = SqliteCacheBackend(db_path)

    pack_cats = _load_pack_names()
    # Build reverse map: object_name -> pack_name
    obj_to_pack = {}
    for pack_name, obj_list in pack_cats.items():
        for obj in obj_list:
            obj_to_pack[obj] = pack_name

    stats = {'tickers': 0, 'objects': 0, 'errors': []}

    if not os.path.isdir(cache_dir):
        return stats

    tickers = list(_iter_ticker_dirs(cache_dir))

    for ticker in tickers:
        ticker_dir = os.path.join(cache_dir, ticker)
        if progress:
            print(f"  Migrating {ticker} …")
        stats['tickers'] += 1

        files = os.listdir(ticker_dir)

        # Determine which files are pack files (annuals.pkl, quarterlys.pkl …)
        pack_files = set(pack_cats.keys())  # {'annuals', 'quarterlys', …}

        for filename in files:
            obj_name = _object_name_from_filename(filename)
            if obj_name is None:
                continue

            filepath = os.path.join(ticker_dir, filename)

            if obj_name in pack_files:
                # This is a grouped pack file — unpack each entry as its own row
                try:
                    with open(filepath, 'rb') as fh:
                        packed = pickle.load(fh)
                    if not isinstance(packed, dict):
                        raise ValueError(f"expected dict, got {type(packed)}")
                    for inner_name, inner_data in packed.items():
                        if not isinstance(inner_data, dict) or 'data' not in inner_data:
                            stats['errors'].append(
                                f"{ticker}/{obj_name}/{inner_name}: unexpected structure"
                            )
                            continue
                        datum = inner_data['data']
                        metadata = inner_data.get('metadata')
                        expiry = inner_data.get('expiry')
                        if not dry_run:
                            backend.store_datum(
                                ticker, inner_name, datum,
                                expiry=expiry,
                                metadata=metadata,
                                pack_name=obj_name,
                            )
                        stats['objects'] += 1
                except Exception as exc:
                    stats['errors'].append(f"{ticker}/{filename}: {exc}")
                    continue
            else:
                # Regular single-object file
                try:
                    raw = yfcm._ReadData(ticker, obj_name)
                    if raw is None:
                        continue
                    datum = raw.get('data') if isinstance(raw, dict) else raw
                    metadata = raw.get('metadata') if isinstance(raw, dict) else None
                    expiry = raw.get('expiry') if isinstance(raw, dict) else None
                    if not dry_run:
                        backend.store_datum(
                            ticker, obj_name, datum,
                            expiry=expiry,
                            metadata=metadata,
                            pack_name=None,
                        )
                    stats['objects'] += 1
                except Exception as exc:
                    stats['errors'].append(f"{ticker}/{filename}: {exc}")
                    continue

    # Migrate _YFC_/ upgrade sentinel files
    yfc_dp = os.path.join(cache_dir, '_YFC_')
    if os.path.isdir(yfc_dp):
        for flag_name in os.listdir(yfc_dp):
            if not dry_run:
                backend.set_upgrade_flag(flag_name)

    backend.close()
    return stats


# -------------------------------------------------------------------------
# migrate_to_files
# -------------------------------------------------------------------------

def migrate_to_files(
    cache_dir: str | None = None,
    progress: bool = True,
    dry_run: bool = False,
) -> dict:
    """
    Read all data from the SQLite cache and write it as files.

    Packed objects (balance_sheet, cashflow …) are reconstructed into
    the correct grouped .pkl files (annuals.pkl, quarterlys.pkl …).

    Parameters
    ----------
    cache_dir : str, optional
        Path to the cache directory. Defaults to the current cache dir.
    progress : bool
        Print per-ticker progress lines.
    dry_run : bool
        If True, read SQLite but do not write files.

    Returns
    -------
    dict with keys 'tickers', 'objects', 'errors'.
    """
    import sqlite3
    import json
    from .yfc_sqlite_manager import _deserialize_datum, _deserialize_metadata, _deserialize_expiry

    if cache_dir is None:
        cache_dir = yfcm.GetCacheDirpath()

    db_path = os.path.join(cache_dir, 'yfc_cache.db')
    if not os.path.isfile(db_path):
        return {'tickers': 0, 'objects': 0, 'errors': ['SQLite DB not found: ' + db_path]}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    stats = {'tickers': 0, 'objects': 0, 'errors': []}

    rows = conn.execute(
        "SELECT ticker, object_name, pack_name, data_blob, data_json, metadata, expiry "
        "FROM cache_data ORDER BY ticker, pack_name NULLS LAST, object_name"
    ).fetchall()

    # Group rows by ticker
    from collections import defaultdict
    by_ticker = defaultdict(list)
    for row in rows:
        by_ticker[row['ticker']].append(row)

    pack_cats = _load_pack_names()

    seen_tickers = set()
    for ticker, ticker_rows in by_ticker.items():
        if ticker not in seen_tickers:
            if progress:
                print(f"  Migrating {ticker} …")
            seen_tickers.add(ticker)
            stats['tickers'] += 1

        # Separate packed vs regular rows
        packed_groups: dict[str, dict] = defaultdict(dict)  # pack_name -> {obj_name -> obj_data}
        regular_rows = []
        for row in ticker_rows:
            if row['pack_name'] is not None:
                packed_groups[row['pack_name']][row['object_name']] = row
            else:
                regular_rows.append(row)

        # Write regular objects
        for row in regular_rows:
            try:
                datum = _deserialize_datum(row['data_blob'], row['data_json'])
                metadata = _deserialize_metadata(row['metadata'])
                expiry = _deserialize_expiry(row['expiry'])
                if datum is None:
                    continue
                if not dry_run:
                    yfcm.StoreCacheDatum(
                        ticker, row['object_name'], datum,
                        expiry=expiry, metadata=metadata,
                    )
                stats['objects'] += 1
            except Exception as exc:
                stats['errors'].append(f"{ticker}/{row['object_name']}: {exc}")

        # Write packed objects — reconstruct the grouped .pkl file
        for pack_name, obj_rows in packed_groups.items():
            pack_fp = os.path.join(cache_dir, ticker, f'{pack_name}.pkl')
            packed_dict = {}
            for obj_name, row in obj_rows.items():
                try:
                    datum = _deserialize_datum(row['data_blob'], row['data_json'])
                    metadata = _deserialize_metadata(row['metadata'])
                    expiry = _deserialize_expiry(row['expiry'])
                    obj_data = {'data': datum}
                    if metadata is not None:
                        obj_data['metadata'] = metadata
                    if expiry is not None:
                        obj_data['expiry'] = expiry
                    packed_dict[obj_name] = obj_data
                    stats['objects'] += 1
                except Exception as exc:
                    stats['errors'].append(f"{ticker}/{pack_name}/{obj_name}: {exc}")

            if packed_dict and not dry_run:
                ticker_dir = os.path.join(cache_dir, ticker)
                os.makedirs(ticker_dir, exist_ok=True)
                try:
                    with open(pack_fp, 'wb') as fh:
                        pickle.dump(packed_dict, fh, 4)
                except Exception as exc:
                    stats['errors'].append(f"{ticker}/{pack_name}.pkl write: {exc}")

    # Reconstruct _YFC_/ sentinel files from cache_meta
    if not dry_run:
        flags = conn.execute(
            "SELECT key FROM cache_meta WHERE key LIKE '_YFC_/%'"
        ).fetchall()
        if flags:
            yfc_dp = os.path.join(cache_dir, '_YFC_')
            os.makedirs(yfc_dp, exist_ok=True)
            for row in flags:
                flag_name = row['key'].removeprefix('_YFC_/')
                flag_fp = os.path.join(yfc_dp, flag_name)
                if not os.path.isfile(flag_fp):
                    open(flag_fp, 'w').close()

    conn.close()
    return stats
