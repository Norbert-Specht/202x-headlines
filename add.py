"""
Manual article submission tool for the 202x Headlines project.

Fetches a URL, extracts Open Graph metadata, and adds the article directly
as accepted — bypassing the scraper and filter stages. Intended for curating
one-off articles that weren't caught by the scrapers.

Duplicate URLs are detected before insertion and produce a clean exit rather
than an error.

Usage:
    python add.py <url>

Example:
    python add.py https://example.com/story-about-something-absurd
"""

import sys
from datetime import datetime, UTC
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from core import get_engine, get_session, init_db
from core.repository import ArticleRepository


# --------------------------------------------------------------------------- #
# Paths and constants                                                          #
# --------------------------------------------------------------------------- #

_PROJECT_ROOT = Path(__file__).resolve().parent
_DB_PATH      = _PROJECT_ROOT / "data" / "articles.db"

# Project ID used to scope all repository queries.
_PROJECT_ID = "202x-headlines"

# Source name recorded in the database for manually submitted articles.
_SOURCE_NAME = "manual"

# Seconds to wait for a page response before giving up.
_REQUEST_TIMEOUT = 15

# Realistic browser User-Agent — some sites block the default requests UA.
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# --------------------------------------------------------------------------- #
# Metadata extraction                                                          #
# --------------------------------------------------------------------------- #

def fetch_metadata(url: str) -> dict:
    """
    Fetch a URL and extract Open Graph metadata from the page.

    Extracts the following fields:
        headline      — og:title (required; exits if missing)
        canonical_url — og:url, falling back to the provided URL
        image_url     — og:image, or None if not present
        date_published — article:published_time parsed as a datetime,
                         falling back to today's UTC date at midnight

    Parameters
    ----------
    url : The URL to fetch and parse.

    Returns
    -------
    dict
        Metadata dict with keys: headline, canonical_url, image_url,
        date_published.

    Raises
    ------
    SystemExit
        If the page cannot be fetched, or if og:title is missing.
    """
    try:
        response = requests.get(
            url,
            headers={"User-Agent": _USER_AGENT},
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Error: could not fetch {url!r}: {exc}")
        sys.exit(1)

    soup = BeautifulSoup(response.text, "html.parser")

    # ------------------------------------------------------------------ #
    # Helper: read a <meta> tag by property or name attribute             #
    # ------------------------------------------------------------------ #

    def get_meta(prop: str) -> str | None:
        """
        Return the content attribute of a <meta> tag matching the given
        property (Open Graph) or name (standard meta) value.

        Parameters
        ----------
        prop : The property or name value to look up, e.g. "og:title".

        Returns
        -------
        str | None
            The tag's content value, stripped of whitespace, or None.
        """
        tag = soup.find("meta", attrs={"property": prop}) or \
              soup.find("meta", attrs={"name": prop})
        if tag and tag.get("content"):
            return tag["content"].strip()
        return None

    # ------------------------------------------------------------------ #
    # og:title — required                                                 #
    # ------------------------------------------------------------------ #

    headline = get_meta("og:title")
    if not headline:
        print("Error: page has no og:title — cannot determine headline.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # og:url — canonical URL, fall back to the URL we fetched             #
    # ------------------------------------------------------------------ #

    canonical_url = get_meta("og:url") or url

    # ------------------------------------------------------------------ #
    # og:image — optional header image                                    #
    # ------------------------------------------------------------------ #

    image_url = get_meta("og:image")

    # ------------------------------------------------------------------ #
    # article:published_time — publication date                           #
    # ------------------------------------------------------------------ #

    raw_date = get_meta("article:published_time")
    date_published = _parse_published_time(raw_date)

    return {
        "headline":       headline,
        "canonical_url":  canonical_url,
        "image_url":      image_url,
        "date_published": date_published,
    }


def _parse_published_time(raw: str | None) -> datetime:
    """
    Parse an article:published_time string into a datetime.

    Accepts ISO 8601 strings (e.g. "2024-01-15T10:30:00Z", "2024-01-15",
    "2024-01-15T10:30:00+02:00"). Returns today's UTC midnight if the value
    is missing or cannot be parsed.

    Parameters
    ----------
    raw : Raw string from the article:published_time meta tag, or None.

    Returns
    -------
    datetime
        A timezone-aware UTC datetime. Falls back to today at midnight UTC.
    """
    if not raw:
        return _today_midnight()

    # Normalise the "Z" suffix — Python's fromisoformat() only handles it
    # from 3.11 onward. Replace it with the equivalent "+00:00" offset.
    normalised = raw.strip().replace("Z", "+00:00")

    try:
        dt = datetime.fromisoformat(normalised)
        # Make timezone-aware if the string had no offset.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except ValueError:
        print(f"  Warning: could not parse date {raw!r} — using today's date.")
        return _today_midnight()


def _today_midnight() -> datetime:
    """
    Return today's date as a timezone-aware UTC datetime at midnight.

    Returns
    -------
    datetime
    """
    today = datetime.now(UTC).date()
    return datetime(today.year, today.month, today.day, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Submission                                                                   #
# --------------------------------------------------------------------------- #

def add(url: str, db_path: Path = _DB_PATH) -> None:
    """
    Fetch metadata for the given URL and add the article directly as accepted.

    Steps:
      1. Fetch the page and extract og:title, og:image, og:url,
         article:published_time.
      2. Check for an existing article with the same URL — exit cleanly if found.
      3. Insert as a candidate, then immediately accept it.
      4. Print a confirmation line.

    Parameters
    ----------
    url    : The article URL to submit.
    db_path : Path to the SQLite database. Defaults to data/articles.db.
    """
    # ------------------------------------------------------------------ #
    # Fetch page metadata                                                  #
    # ------------------------------------------------------------------ #

    print(f"Fetching {url} …")
    meta = fetch_metadata(url)

    headline       = meta["headline"]
    canonical_url  = meta["canonical_url"]
    image_url      = meta["image_url"]
    date_published = meta["date_published"]

    # ------------------------------------------------------------------ #
    # Set up database                                                      #
    # ------------------------------------------------------------------ #

    engine = get_engine(db_path=db_path)
    init_db(engine)

    # ------------------------------------------------------------------ #
    # Duplicate check                                                      #
    # ------------------------------------------------------------------ #

    with get_session(engine) as session:
        repo = ArticleRepository(session)

        # Check both the canonical URL from og:url and the original input URL,
        # since either may already be in the database from a previous scrape run.
        for check_url in {canonical_url, url}:
            existing = repo.get_article_by_url(source_url=check_url)
            if existing is not None:
                print(f"Already in database (id={existing.id}, status={existing.status!r}):")
                print(f"  {existing.headline}")
                return

        # ------------------------------------------------------------------ #
        # Insert as candidate, then accept immediately                        #
        # ------------------------------------------------------------------ #

        # add_candidate() creates the article in "candidate" status; accept()
        # transitions it to "accepted" and sets date_published / publication_year.
        # Using both methods rather than setting status directly ensures all
        # business logic (review records, publication_year extraction) runs correctly.
        article = repo.add_candidate(
            project_id=_PROJECT_ID,
            headline=headline,
            source_url=canonical_url,
            source_name=_SOURCE_NAME,
            excerpt=None,
            date_scraped=datetime.now(UTC),
            image_url=image_url,
        )
        repo.accept(
            article_id=article.id,
            date_published=date_published,
        )

    # ------------------------------------------------------------------ #
    # Confirmation                                                         #
    # ------------------------------------------------------------------ #

    image_note = f"image found ({image_url})" if image_url else "no image"
    print(f"\nAdded:")
    print(f"  Headline : {headline}")
    print(f"  URL      : {canonical_url}")
    print(f"  Date     : {date_published.strftime('%Y-%m-%d')}")
    print(f"  Image    : {image_note}")


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add.py <url>")
        sys.exit(1)

    add(url=sys.argv[1])
