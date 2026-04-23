"""
Tests for scraper.py.

All HTTP calls are mocked — no real network requests are made.
Tests cover config loading, feed parsing, and the full scrape orchestration.
"""

import textwrap
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core import get_engine, get_session, init_db
from core.repository import ArticleRepository
from scraper import fetch_feed, load_sources, scrape


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _make_feed_entry(title="Test Headline", link="http://example.com/1", summary="A short summary."):
    """Return a minimal feedparser-style entry dict."""
    entry = MagicMock()
    entry.get = lambda key, default="": {
        "title": title,
        "link": link,
        "summary": summary,
    }.get(key, default)
    return entry


def _make_parsed_feed(entries, bozo=False):
    """Return a minimal feedparser result object."""
    feed = MagicMock()
    feed.entries = entries
    feed.bozo = bozo
    # Only set bozo_exception when bozo is True, mirroring feedparser behaviour.
    if not bozo:
        del feed.bozo_exception
    return feed


# --------------------------------------------------------------------------- #
# load_sources tests                                                           #
# --------------------------------------------------------------------------- #

def test_load_sources_parses_valid_yaml(tmp_path):
    """load_sources should return a dict with project_id and sources keys."""
    config_file = tmp_path / "sources.yaml"
    config_file.write_text(textwrap.dedent("""
        project_id: test-project
        sources:
          - name: Test Feed
            url: http://example.com/rss
            enabled: true
    """))

    config = load_sources(config_path=config_file)

    assert config["project_id"] == "test-project"
    assert len(config["sources"]) == 1
    assert config["sources"][0]["name"] == "Test Feed"
    assert config["sources"][0]["enabled"] is True


def test_load_sources_raises_on_missing_file(tmp_path):
    """load_sources should raise FileNotFoundError for a non-existent path."""
    missing = tmp_path / "does_not_exist.yaml"

    with pytest.raises(FileNotFoundError, match="Sources config not found"):
        load_sources(config_path=missing)


# --------------------------------------------------------------------------- #
# fetch_feed tests                                                             #
# --------------------------------------------------------------------------- #

def test_fetch_feed_returns_two_entries_for_two_item_feed():
    """
    A mocked feed with two entries should return two correctly structured dicts,
    each with headline, source_url, excerpt, and date_scraped.
    """
    entries = [
        _make_feed_entry(title="Headline One", link="http://example.com/1", summary="Summary one."),
        _make_feed_entry(title="Headline Two", link="http://example.com/2", summary="Summary two."),
    ]
    mock_feed = _make_parsed_feed(entries=entries)

    with patch("scraper.feedparser.parse", return_value=mock_feed):
        results = fetch_feed(source={"name": "Test Feed", "url": "http://example.com/rss"})

    assert len(results) == 2

    assert results[0]["headline"] == "Headline One"
    assert results[0]["source_url"] == "http://example.com/1"
    assert results[0]["excerpt"] == "Summary one."
    assert isinstance(results[0]["date_scraped"], datetime)

    assert results[1]["headline"] == "Headline Two"
    assert results[1]["source_url"] == "http://example.com/2"


def test_fetch_feed_skips_entry_with_no_url():
    """An entry with an empty link should be excluded from results."""
    entries = [
        _make_feed_entry(title="Has URL", link="http://example.com/1"),
        _make_feed_entry(title="No URL",  link=""),
    ]
    mock_feed = _make_parsed_feed(entries=entries)

    with patch("scraper.feedparser.parse", return_value=mock_feed):
        results = fetch_feed(source={"name": "Test Feed", "url": "http://example.com/rss"})

    assert len(results) == 1
    assert results[0]["source_url"] == "http://example.com/1"


def test_fetch_feed_returns_empty_list_on_network_error():
    """A network-level exception should be caught and return an empty list."""
    with patch("scraper.feedparser.parse", side_effect=OSError("connection refused")):
        results = fetch_feed(source={"name": "Bad Feed", "url": "http://unreachable.example.com/rss"})

    assert results == []


# --------------------------------------------------------------------------- #
# scrape orchestration tests                                                   #
# --------------------------------------------------------------------------- #

def test_scrape_adds_new_and_skips_duplicates(tmp_path):
    """
    scrape() should add articles whose URLs are not in the database,
    skip articles whose URLs are already present, and return an accurate
    summary dict.

    Two sources, three entries total:
      - Source A: two entries, one already in the database (skipped)
      - Source B: one new entry (added)
    Expected: fetched=3, added=2, skipped=1, errors=0
    """
    # ------------------------------------------------------------------ #
    # Build a real in-memory database with one pre-existing article        #
    # ------------------------------------------------------------------ #
    engine = get_engine(db_path=":memory:")
    init_db(engine)

    with get_session(engine) as session:
        repo = ArticleRepository(session)
        repo.add_candidate(
            project_id="test-project",
            headline="Already Stored",
            source_url="http://example.com/already",
            source_name="Test Feed A",
            excerpt=None,
            date_scraped=datetime(2025, 1, 1),
        )

    # ------------------------------------------------------------------ #
    # Write a minimal sources.yaml for this test                           #
    # ------------------------------------------------------------------ #
    config_file = tmp_path / "sources.yaml"
    config_file.write_text(textwrap.dedent("""
        project_id: test-project
        sources:
          - name: Feed A
            url: http://example.com/feed-a
            enabled: true
          - name: Feed B
            url: http://example.com/feed-b
            enabled: true
    """))

    # ------------------------------------------------------------------ #
    # Mock fetch_feed to return controlled entries per source              #
    # ------------------------------------------------------------------ #
    feed_a_entries = [
        {"headline": "New from A",       "source_url": "http://example.com/new-a",    "excerpt": None, "date_scraped": datetime(2025, 6, 1)},
        {"headline": "Already Stored",   "source_url": "http://example.com/already",  "excerpt": None, "date_scraped": datetime(2025, 6, 1)},
    ]
    feed_b_entries = [
        {"headline": "New from B",       "source_url": "http://example.com/new-b",    "excerpt": None, "date_scraped": datetime(2025, 6, 1)},
    ]

    def fake_fetch_feed(source):
        if "feed-a" in source["url"]:
            return feed_a_entries
        return feed_b_entries

    # Also mock get_engine so scrape() uses our in-memory database.
    with patch("scraper.fetch_feed", side_effect=fake_fetch_feed), \
         patch("scraper.get_engine", return_value=engine), \
         patch("scraper.init_db"):

        summary = scrape(config_path=config_file, db_path=":memory:")

    assert summary["fetched"] == 3
    assert summary["added"] == 2
    assert summary["skipped"] == 1
    assert summary["errors"] == 0

    # Verify the two new articles are actually in the database.
    # Collect source_urls inside the session — ORM objects become detached
    # once the session closes and attribute access raises DetachedInstanceError.
    with get_session(engine) as session:
        repo = ArticleRepository(session)
        candidates = repo.get_candidates(project_id="test-project")
        urls = {a.source_url for a in candidates}
        count = len(candidates)

    # Three total: the one pre-inserted + two new ones.
    assert count == 3
    assert "http://example.com/new-a" in urls
    assert "http://example.com/new-b" in urls
