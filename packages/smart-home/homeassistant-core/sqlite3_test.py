print("testing sqlite3...")

import os
import sqlite3
from packaging.version import Version


SQLITE_VERSION = os.getenv('SQLITE_VERSION', None)

current_version = Version(sqlite3.sqlite_version)
required_version = Version(SQLITE_VERSION)

if not SQLITE_VERSION:
    raise AssertionError(
        f"SQLITE_VERSION env variable is not set!"
    )

print(f'sqlite3 version: {current_version} [required: {required_version}]')

if current_version < required_version:
    raise AssertionError(
        f"SQLite version is {current_version}, but {required_version} is required."
    )

print("sqlite3 ok")