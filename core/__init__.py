"""
Database initialisation and session management for the article tracker.

Usage
-----
    from core import get_engine, init_db, get_session

    engine = get_engine()          # defaults to data/articles.db
    init_db(engine)                # creates tables if they don't exist

    with get_session(engine) as session:
        repo = ArticleRepository(session)
        ...
"""

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session

from core.models import Base

# Default database path relative to the project root.
_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "articles.db"


def get_engine(db_path: str | Path | None = None) -> Engine:
    """
    Create and return a SQLAlchemy engine for the given SQLite path.

    Creates the data/ directory automatically if it doesn't exist, so
    callers never need to mkdir before calling this.

    Parameters
    ----------
    db_path : Path to the SQLite file. Defaults to data/articles.db
              relative to the project root. Pass ":memory:" for in-memory
              use (useful in tests).

    Returns
    -------
    Engine
    """
    if db_path is None:
        db_path = _DEFAULT_DB_PATH

    path = Path(db_path)

    # ":memory:" is a SQLite special value — don't try to create it as a dir.
    if str(path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
        connect_url = f"sqlite:///{path}"
    else:
        connect_url = "sqlite:///:memory:"

    return create_engine(connect_url)


def init_db(engine: Engine) -> None:
    """
    Create all tables defined in the ORM models if they don't exist yet.

    Safe to call on every startup — SQLAlchemy only creates tables that are
    missing, it never drops or alters existing ones.

    Parameters
    ----------
    engine : An Engine returned by get_engine().
    """
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session(engine: Engine):
    """
    Context manager that yields a SQLAlchemy Session with automatic
    commit/rollback/close handling.

    Commits on clean exit, rolls back on any exception, and always closes
    the session.

    Parameters
    ----------
    engine : An Engine returned by get_engine().

    Yields
    ------
    Session
    """
    SessionFactory = sessionmaker(bind=engine)
    session: Session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
