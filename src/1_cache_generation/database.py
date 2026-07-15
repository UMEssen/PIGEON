"""
Stage 1 -- Database Connection Layer
=====================================

Provides a thin wrapper around psycopg (PostgreSQL driver) for connecting
to the FHIR metrics database and executing the SQL queries defined in
``queries.py``.

WHY A WRAPPER CLASS?
--------------------
The download script uses ThreadPoolExecutor -- each thread needs its own
database connection (psycopg connections are NOT thread-safe).  Having a
small factory-friendly class makes it easy to spin up a connection per
worker thread and tear it down when done.

CONFIGURATION
-------------
All connection parameters default to the values in ``config.py`` (which
reads from environment variables / ``.env``).  You can override any
parameter at instantiation time for testing or pointing at a different
database.

USAGE
-----
    from database import MetricsMiner

    miner = MetricsMiner()                       # uses config.py defaults
    rows, columns = miner.execute_query("SELECT 1;")
    miner.close()

    # Or with a context manager:
    with MetricsMiner() as miner:
        rows, cols = miner.execute_query(query)
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project-level config importable no matter where this file lives.
# config.py sits two directories up from src/1_cache_generation/.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import METRICS_DB, METRICS_USER, METRICS_PASSWORD, METRICS_HOST, METRICS_PORT

try:
    import psycopg
except ImportError:
    raise ImportError(
        "psycopg (v3) is required.  Install it with:  pip install 'psycopg[binary]'"
    )


class MetricsMiner:
    """Lightweight database connection wrapper for FHIR metrics queries.

    Each instance holds exactly one psycopg connection + cursor pair.
    Create one per thread; do NOT share across threads.

    Parameters
    ----------
    dbname : str, optional
        Database name.  Defaults to ``config.METRICS_DB``.
    user : str, optional
        Database user.  Defaults to ``config.METRICS_USER``.
    password : str, optional
        Database password.  Defaults to ``config.METRICS_PASSWORD``.
    host : str, optional
        Database host.  Defaults to ``config.METRICS_HOST``.
    port : int, optional
        Database port.  Defaults to ``config.METRICS_PORT``.
    """

    def __init__(
        self,
        dbname: str | None = None,
        user: str | None = None,
        password: str | None = None,
        host: str | None = None,
        port: int | None = None,
    ):
        # Fall back to config.py values when a parameter is not provided.
        # This lets callers override individual params for testing while
        # keeping everything else at the project default.
        self._dbname = dbname or METRICS_DB
        self._user = user or METRICS_USER
        self._password = password or METRICS_PASSWORD
        self._host = host or METRICS_HOST
        self._port = port or METRICS_PORT

        self.connection = psycopg.connect(
            f"dbname={self._dbname} "
            f"user={self._user} "
            f"password={self._password} "
            f"host={self._host} "
            f"port={self._port}"
        )
        self.cursor = self.connection.cursor()

    # ------------------------------------------------------------------
    # Context-manager support  (with MetricsMiner() as m: ...)
    # ------------------------------------------------------------------

    def __enter__(self) -> MetricsMiner:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    def execute_query(self, query: str) -> tuple[list, list[str]]:
        """Execute a SQL query and return all results.

        Parameters
        ----------
        query : str
            A complete SQL query string (placeholders like ``(batches)``
            must already be replaced with real values).

        Returns
        -------
        rows : list[tuple]
            All result rows from fetchall().
        column_names : list[str]
            Column names in the same order as the tuple elements.

        Raises
        ------
        Exception
            Re-raises any database error after attempting a rollback if
            the connection was lost.
        """
        try:
            self.cursor.execute(query)
            column_names = [desc[0] for desc in self.cursor.description]
            return self.cursor.fetchall(), column_names
        except Exception as exc:
            # If the connection dropped, rollback so the next call has a
            # chance of working (or the caller can reconnect).
            if "connection" in str(exc).lower() or "closed" in str(exc).lower():
                try:
                    self.connection.rollback()
                except Exception:
                    pass  # already broken -- nothing we can do
            raise

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close cursor and connection.  Safe to call multiple times."""
        try:
            self.cursor.close()
        except Exception:
            pass
        try:
            self.connection.close()
        except Exception:
            pass
