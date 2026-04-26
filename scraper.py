"""
RSS scraper for the 202x Headlines pipeline.

Reads a list of RSS sources from sources.yaml, fetches each feed, and
stores new entries as candidate articles in the database. Safe to run
repeatedly — already-seen URLs are skipped without overwriting anything.

Usage:
    python scraper.py
"""

import warnings
from datetime import datetime, UTC
from pathlib import Path

import feedparser
import yaml
from bs4 import BeautifulSoup

from core import get_engine, get_session, init_db
from core.repository import ArticleRepository


# --------------------------------------------------------------------------- #
# Paths                                                                        #
# --------------------------------------------------------------------------- #

# Both paths are relative to the project root (the directory this file lives in).
_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_SOURCES_PATH = _PROJECT_ROOT / "sources.yaml"
_DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "articles.db"

# Width of the source-name column in the printed run summary.
_NAME_COL_WIDTH = 20

# Maximum length of the excerpt stored per article.
_EXCERPT_MAX_CHARS = 300


# --------------------------------------------------------------------------- #
# Config loading                                                               #
# --------------------------------------------------------------------------- #

def load_sources(config_path: str | Path) -> dict:
    """
    Read and parse the sources YAML config file.

    Expected structure:
        project_id: <string>
        sources:
          - name: <string>
            url:  <string>
            enabled: <bool>

    Parameters
    ----------
    config_path : Path to the sources.yaml file.

    Returns
    -------
    dict
        The full parsed config, e.g. {"project_id": "...", "sources": [...]}.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the given path.
    ValueError
        If the file exists but cannot be parsed as valid YAML, or is missing
        the required top-level keys.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Sources config not found: {path}\n"
            "Create sources.yaml in the project root before running the scraper."
        )

    with path.open("r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Could not parse {path}: {exc}") from exc

    # Validate that the required keys are present.
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level.")
    if "project_id" not in config:
        raise ValueError(f"{path} is missing the required 'project_id' key.")
    if "sources" not in config:
        raise ValueError(f"{path} is missing the required 'sources' key.")

    return config


# --------------------------------------------------------------------------- #
# Feed fetching                                                                #
# --------------------------------------------------------------------------- #

def _strip_html(raw: str) -> str:
    """
    Remove HTML tags from a string and return plain text.

    Parameters
    ----------
    raw : A string that may contain HTML markup.

    Returns
    -------
    str
        Plain text with all HTML tags removed and whitespace normalised.
    """
    # BeautifulSoup handles malformed HTML gracefully — feedparser summaries
    # often contain partial tags or entities.
    text = BeautifulSoup(raw, "html.parser").get_text(separator=" ")
    # Collapse runs of whitespace left behind by block-level tag removal.
    return " ".join(text.split())


def _extract_image_url(entry) -> str | None:
    """
    Extract a header image URL from an RSS feed entry.

    Checks three common RSS media extension fields in priority order:
      1. media_content  — explicit image metadata (Media RSS extension)
      2. enclosures     — standard RSS 2.0 binary attachment mechanism
      3. media_thumbnail — lower-resolution preview image

    Only http/https URLs are accepted. The first valid URL found is returned.

    Parameters
    ----------
    entry : A feedparser entry object (supports .get() dict-style access).

    Returns
    -------
    str | None
        The first valid image URL found, or None if none is present.
    """
    # media_content is a list of dicts; each may have 'url', 'type', 'medium'.
    for item in entry.get("media_content", []):
        url = item.get("url", "").strip()
        if url.startswith("http"):
            return url

    # enclosures is the standard RSS 2.0 attachment list; each item has
    # 'href', 'type', and 'length'. Only accept explicitly image-typed entries
    # so we don't accidentally store audio or video URLs as image_url.
    for enc in entry.get("enclosures", []):
        url = (enc.get("href") or enc.get("url") or "").strip()
        mime = enc.get("type", "")
        if url.startswith("http") and mime.startswith("image/"):
            return url

    # media_thumbnail is a list of dicts with a 'url' key — lower priority
    # than media_content but widely used as a feed preview image.
    for item in entry.get("media_thumbnail", []):
        url = item.get("url", "").strip()
        if url.startswith("http"):
            return url

    return None


def fetch_feed(source: dict) -> list[dict]:
    """
    Fetch and parse the RSS feed for a single source.

    Each returned dict represents one feed entry and contains:
        headline    (str)           — the entry title
        source_url  (str)           — the entry link
        excerpt     (str | None)    — first 300 chars of summary, HTML stripped
        image_url   (str | None)    — header image URL if present in the feed
        date_scraped (datetime)     — UTC time of this scrape run (not pub date)

    Entries with no URL are silently skipped.
    Network or parse errors are caught and logged; an empty list is returned
    so one bad source does not abort the whole run.

    Parameters
    ----------
    source : A dict with at least "name" and "url" keys (from sources.yaml).

    Returns
    -------
    list[dict]
        One dict per valid feed entry. Empty list on error.
    """
    name = source.get("name", source.get("url", "unknown"))
    url = source["url"]

    try:
        # feedparser raises no exceptions itself — errors appear in feed.bozo
        # and feed.bozo_exception. We check these below.
        feed = feedparser.parse(url)
    except Exception as exc:
        print(f"  WARNING: Could not fetch feed for {name!r}: {exc}")
        return []

    # feedparser sets bozo=True for malformed XML, but often still parses
    # partial content. Only treat it as a full failure if we got no entries.
    if feed.bozo and not feed.entries:
        exc = getattr(feed, "bozo_exception", "unknown error")
        print(f"  WARNING: Malformed feed for {name!r}: {exc}")
        return []

    now = datetime.now(UTC)
    entries = []

    for entry in feed.entries:
        # ------------------------------------------------------------------ #
        # Extract URL — skip entries that have none                           #
        # ------------------------------------------------------------------ #
        source_url = entry.get("link", "").strip()
        if not source_url:
            continue

        # ------------------------------------------------------------------ #
        # Extract headline                                                    #
        # ------------------------------------------------------------------ #
        headline = entry.get("title", "").strip()
        if not headline:
            continue

        # ------------------------------------------------------------------ #
        # Extract and truncate excerpt                                        #
        # ------------------------------------------------------------------ #
        raw_summary = entry.get("summary", "")
        if raw_summary:
            clean = _strip_html(raw_summary)
            # Truncate to the limit; strip any dangling partial word.
            excerpt = clean[:_EXCERPT_MAX_CHARS] if len(clean) <= _EXCERPT_MAX_CHARS else clean[:_EXCERPT_MAX_CHARS].rsplit(" ", 1)[0]
        else:
            excerpt = None

        entries.append({
            "headline": headline,
            "source_url": source_url,
            "excerpt": excerpt,
            "image_url": _extract_image_url(entry),
            # Use the current UTC time, not the feed's published date, so we
            # know when this article entered our pipeline.
            "date_scraped": now,
        })

    return entries


# --------------------------------------------------------------------------- #
# Scrape orchestration                                                         #
# --------------------------------------------------------------------------- #

def scrape(
    config_path: str | Path = _DEFAULT_SOURCES_PATH,
    db_path: str | Path = _DEFAULT_DB_PATH,
) -> dict:
    """
    Run a full scrape across all enabled sources in the config.

    For each feed entry:
      - If the URL already exists in the database, it is skipped.
      - If the URL is new, it is stored as a candidate article.

    Prints a per-source summary line and a totals line when complete.

    Parameters
    ----------
    config_path : Path to sources.yaml. Defaults to sources.yaml in the
                  project root.
    db_path     : Path to the SQLite database file. Defaults to
                  data/articles.db in the project root.

    Returns
    -------
    dict
        Run summary with keys: "fetched", "added", "skipped", "errors".
    """
    # ------------------------------------------------------------------ #
    # Load config                                                          #
    # ------------------------------------------------------------------ #

    config = load_sources(config_path)
    project_id = config["project_id"]
    sources = [s for s in config["sources"] if s.get("enabled", True)]

    # ------------------------------------------------------------------ #
    # Set up database                                                      #
    # ------------------------------------------------------------------ #

    engine = get_engine(db_path=db_path)
    init_db(engine)

    # ------------------------------------------------------------------ #
    # Print run header                                                     #
    # ------------------------------------------------------------------ #

    title = project_id.replace("-", " ").title()
    divider = "─" * 43
    print(f"\n{title} — Scrape Run")
    print(divider)

    # ------------------------------------------------------------------ #
    # Process each source                                                  #
    # ------------------------------------------------------------------ #

    total_fetched = 0
    total_added = 0
    total_skipped = 0
    total_errors = 0

    for source in sources:
        name = source.get("name", source["url"])
        entries = fetch_feed(source)

        # fetch_feed returns [] on error and prints its own warning; treat
        # an empty result with no entries in the feed as a possible error.
        if not entries and _feed_had_error(source):
            total_errors += 1
            print(f"  {name:<{_NAME_COL_WIDTH}} ERROR — could not fetch feed")
            continue

        source_fetched = len(entries)
        source_added = 0
        source_skipped = 0

        with get_session(engine) as session:
            repo = ArticleRepository(session)

            for entry in entries:
                # ---------------------------------------------------------- #
                # Duplicate check — skip if already in the database           #
                # ---------------------------------------------------------- #
                existing = repo.get_article_by_url(source_url=entry["source_url"])
                if existing is not None:
                    source_skipped += 1
                    continue

                # ---------------------------------------------------------- #
                # Insert new candidate                                         #
                # ---------------------------------------------------------- #
                repo.add_candidate(
                    project_id=project_id,
                    headline=entry["headline"],
                    source_url=entry["source_url"],
                    source_name=name,
                    excerpt=entry["excerpt"],
                    date_scraped=entry["date_scraped"],
                    image_url=entry.get("image_url"),
                )
                source_added += 1

        total_fetched += source_fetched
        total_added += source_added
        total_skipped += source_skipped

        print(
            f"  {name:<{_NAME_COL_WIDTH}} "
            f"fetched {source_fetched:<5} "
            f"added {source_added:<5} "
            f"skipped {source_skipped}"
        )

    # ------------------------------------------------------------------ #
    # Print totals                                                         #
    # ------------------------------------------------------------------ #

    print(divider)
    print(
        f"  Total: fetched {total_fetched}   "
        f"added {total_added}   "
        f"skipped {total_skipped}   "
        f"errors {total_errors}"
    )
    print()

    return {
        "fetched": total_fetched,
        "added": total_added,
        "skipped": total_skipped,
        "errors": total_errors,
    }


def _feed_had_error(source: dict) -> bool:
    """
    Re-probe whether a feed genuinely failed (no entries and no network content).

    fetch_feed already prints warnings for errors; this helper lets scrape()
    decide whether an empty result is an error or just an empty feed.

    Parameters
    ----------
    source : The source dict from sources.yaml.

    Returns
    -------
    bool
        True if the feed appears to have failed, False if it may just be empty.
    """
    try:
        feed = feedparser.parse(source["url"])
        # bozo with no entries and a bozo_exception is a genuine failure.
        return bool(feed.bozo and not feed.entries and hasattr(feed, "bozo_exception"))
    except Exception:
        return True


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    scrape()
