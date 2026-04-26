"""
Migration: add image_url column to the articles table.

Safe to run on a database that already contains articles — checks whether
the column exists before attempting to add it, so running this script a
second time is harmless.

Usage:
    python migrate_add_image_url.py
"""

from pathlib import Path

from sqlalchemy import inspect, text

from core import get_engine, init_db


# Default database path — matches the path used by the rest of the project.
_DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "articles.db"

# Single column to add. Defined as a list to mirror the pattern established
# by migrate_add_filter_fields.py and to make future additions straightforward.
_COLUMNS_TO_ADD = [
    ("image_url", "VARCHAR"),
]


def migrate(db_path: str | Path = _DEFAULT_DB_PATH) -> None:
    """
    Add the image_url column to the articles table.

    Uses raw ALTER TABLE SQL rather than Alembic to keep the project
    dependency-free for simple schema changes. SQLite supports adding
    columns with ALTER TABLE but not removing or altering them, which is
    sufficient here.

    Parameters
    ----------
    db_path : Path to the SQLite database file. Defaults to data/articles.db.
    """
    engine = get_engine(db_path=db_path)

    # ------------------------------------------------------------------ #
    # Ensure the database and base tables exist                            #
    # ------------------------------------------------------------------ #

    # init_db is a no-op on tables that already exist, so this is safe to
    # call on both fresh and populated databases.
    init_db(engine)

    # ------------------------------------------------------------------ #
    # Inspect current schema                                               #
    # ------------------------------------------------------------------ #

    inspector = inspect(engine)
    existing_columns = {col["name"] for col in inspector.get_columns("articles")}

    # ------------------------------------------------------------------ #
    # Add missing columns                                                  #
    # ------------------------------------------------------------------ #

    with engine.connect() as conn:
        for column_name, column_type in _COLUMNS_TO_ADD:
            if column_name in existing_columns:
                print(f"  Column '{column_name}' already exists — skipping.")
            else:
                # SQLite ALTER TABLE ADD COLUMN syntax.
                sql = f"ALTER TABLE articles ADD COLUMN {column_name} {column_type}"
                conn.execute(text(sql))
                print(f"  Added column '{column_name}' ({column_type}).")

        conn.commit()

    print("\nMigration complete.")


if __name__ == "__main__":
    migrate()
