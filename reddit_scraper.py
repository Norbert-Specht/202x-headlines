"""
Reddit scraper for the 202x Headlines pipeline.

Reads reddit_sources from sources.yaml, fetches top posts from the past 24h
via Reddit's public JSON API (no credentials required), and stores new entries
as candidate articles in the database. Safe to run repeatedly — already-seen
URLs are skipped without overwriting anything.

Usage:
    python reddit_scraper.py
"""

from datetime import datetime, UTC
from pathlib import Path

import requests
import yaml

from core import get_engine, get_session, init_db
from core.repository import ArticleRepository


# --------------------------------------------------------------------------- #
# Paths and constants                                                          #
# --------------------------------------------------------------------------- #

# Both paths are relative to the project root (the directory this file lives in).
_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_SOURCES_PATH = _PROJECT_ROOT / "sources.yaml"
_DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "articles.db"

# Width of the source-name column in the printed run summary.
_NAME_COL_WIDTH = 20

# Reddit's public JSON API endpoint — no OAuth required, rate-limited by IP.
_REDDIT_JSON_URL = "https://www.reddit.com/r/{subreddit}/top.json"

# Reddit blocks the default `python-requests` User-Agent with a 429/403.
# A descriptive string with contact info is the format Reddit recommends.
_USER_AGENT = "202x-headlines-scraper/1.0 (automated research tool; contact: research)"

# How many top posts to request per fetch. Reddit's hard maximum is 100.
_FETCH_LIMIT = 100

# Reddit `t` parameter — limits results to the top posts within the past day.
_TIME_FILTER = "day"

# Seconds to wait for a response before giving up on a request.
_REQUEST_TIMEOUT = 15


# --------------------------------------------------------------------------- #
# Config loading                                                               #
# --------------------------------------------------------------------------- #

def load_reddit_sources(config_path: str | Path) -> dict:
    """
    Read sources.yaml and return the project_id and reddit_sources list.

    Expected structure in sources.yaml:
        project_id: <string>
        reddit_sources:
          - name: <string>
            subreddit: <string>
            enabled: <bool>

    Parameters
    ----------
    config_path : Path to the sources.yaml file.

    Returns
    -------
    dict
        A dict with keys "project_id" and "reddit_sources" (list of source dicts).

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the given path.
    ValueError
        If the file cannot be parsed as valid YAML, is missing project_id,
        or has no reddit_sources key.
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

    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level.")
    if "project_id" not in config:
        raise ValueError(f"{path} is missing the required 'project_id' key.")
    if "reddit_sources" not in config:
        raise ValueError(f"{path} is missing the required 'reddit_sources' key.")

    return {
        "project_id": config["project_id"],
        "reddit_sources": config["reddit_sources"],
    }


# --------------------------------------------------------------------------- #
# Post fetching                                                                #
# --------------------------------------------------------------------------- #

def fetch_reddit_posts(source: dict) -> list[dict]:
    """
    Fetch top posts from a subreddit via Reddit's public JSON API.

    Requests the top posts from the past 24 hours (t=day) using the public
    .json endpoint — no API credentials are required. Self-posts and posts
    that link back to reddit.com itself are skipped, since we want external
    article URLs only.

    Each returned dict contains:
        headline    (str)       — the post title
        source_url  (str)       — the external article URL (not the Reddit permalink)
        excerpt     (str | None)— always None for link posts (no body text)
        image_url   (str | None)— thumbnail URL if Reddit provided a real one
        date_scraped (datetime) — UTC time of this scrape run

    Network and HTTP errors are caught and logged; an empty list is returned
    so one bad source does not abort the whole run.

    Parameters
    ----------
    source : A dict with at least "name" and "subreddit" keys (from sources.yaml).

    Returns
    -------
    list[dict]
        One dict per valid post. Empty list on error.
    """
    name = source.get("name", source.get("subreddit", "unknown"))
    subreddit = source["subreddit"]
    url = _REDDIT_JSON_URL.format(subreddit=subreddit)

    try:
        response = requests.get(
            url,
            params={"t": _TIME_FILTER, "limit": _FETCH_LIMIT},
            headers={"User-Agent": _USER_AGENT},
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        print(f"  WARNING: Could not fetch Reddit posts for {name!r}: {exc}")
        return []
    except ValueError as exc:
        # response.json() raises ValueError on malformed JSON
        print(f"  WARNING: Could not parse Reddit response for {name!r}: {exc}")
        return []

    children = data.get("data", {}).get("children", [])
    now = datetime.now(UTC)
    posts = []

    for child in children:
        post = child.get("data", {})

        # ------------------------------------------------------------------ #
        # Extract and validate the external article URL                       #
        # ------------------------------------------------------------------ #
        source_url = post.get("url", "").strip()
        if not source_url:
            continue

        # Skip self-posts and posts that link to reddit.com — these are not
        # external articles and would pollute the candidate queue.
        if "reddit.com" in source_url:
            continue

        # ------------------------------------------------------------------ #
        # Extract headline                                                    #
        # ------------------------------------------------------------------ #
        headline = post.get("title", "").strip()
        if not headline:
            continue

        # ------------------------------------------------------------------ #
        # Extract thumbnail image URL                                        #
        # ------------------------------------------------------------------ #
        # Reddit populates 'thumbnail' with a real URL, or with placeholder
        # strings like 'self', 'default', 'nsfw', or 'spoiler'. Only use it
        # if it looks like a real HTTP URL.
        raw_thumbnail = post.get("thumbnail", "")
        image_url = raw_thumbnail if raw_thumbnail.startswith("http") else None

        posts.append({
            "headline": headline,
            "source_url": source_url,
            # r/nottheonion is a link-only subreddit — selftext is always empty.
            # Storing None keeps the schema consistent with RSS candidates.
            "excerpt": None,
            "image_url": image_url,
            # Use current UTC time (not Reddit's created_utc) so we know when
            # this article entered our pipeline, consistent with the RSS scraper.
            "date_scraped": now,
        })

    return posts


# --------------------------------------------------------------------------- #
# Scrape orchestration                                                         #
# --------------------------------------------------------------------------- #

def scrape_reddit(
    config_path: str | Path = _DEFAULT_SOURCES_PATH,
    db_path: str | Path = _DEFAULT_DB_PATH,
) -> dict:
    """
    Run a full Reddit scrape across all enabled reddit_sources in the config.

    For each post:
      - If the article URL already exists in the database, it is skipped.
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

    config = load_reddit_sources(config_path)
    project_id = config["project_id"]
    sources = [s for s in config["reddit_sources"] if s.get("enabled", True)]

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
    print(f"\n{title} — Reddit Scrape Run")
    print(divider)

    # ------------------------------------------------------------------ #
    # Process each source                                                  #
    # ------------------------------------------------------------------ #

    total_fetched = 0
    total_added = 0
    total_skipped = 0
    total_errors = 0

    for source in sources:
        name = source.get("name", source.get("subreddit", "unknown"))
        posts = fetch_reddit_posts(source)

        if not posts:
            total_errors += 1
            print(f"  {name:<{_NAME_COL_WIDTH}} ERROR — could not fetch posts")
            continue

        source_fetched = len(posts)
        source_added = 0
        source_skipped = 0

        with get_session(engine) as session:
            repo = ArticleRepository(session)

            for post in posts:
                # ---------------------------------------------------------- #
                # Duplicate check — skip if URL is already in the database    #
                # ---------------------------------------------------------- #
                existing = repo.get_article_by_url(source_url=post["source_url"])
                if existing is not None:
                    source_skipped += 1
                    continue

                # ---------------------------------------------------------- #
                # Insert new candidate                                         #
                # ---------------------------------------------------------- #
                repo.add_candidate(
                    project_id=project_id,
                    headline=post["headline"],
                    source_url=post["source_url"],
                    source_name=name,
                    excerpt=post["excerpt"],
                    date_scraped=post["date_scraped"],
                    image_url=post.get("image_url"),
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


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    scrape_reddit()
