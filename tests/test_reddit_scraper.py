"""
Tests for reddit_scraper.py.

All HTTP calls are mocked — no real network requests are made.
Tests cover config loading, post fetching, and full scrape orchestration.
"""

import textwrap
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core import get_engine, get_session, init_db
from core.repository import ArticleRepository
from reddit_scraper import fetch_reddit_posts, load_reddit_sources, scrape_reddit


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _make_reddit_response(posts: list[dict]) -> MagicMock:
    """
    Build a mock requests.Response whose .json() returns the Reddit listing shape.

    Parameters
    ----------
    posts : List of post-data dicts, each with at least "title" and "url".

    Returns
    -------
    MagicMock
        A mock that satisfies the raise_for_status() + .json() contract.
    """
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "data": {
            "children": [{"data": post} for post in posts]
        }
    }
    return mock


# --------------------------------------------------------------------------- #
# load_reddit_sources tests                                                    #
# --------------------------------------------------------------------------- #

def test_load_reddit_sources_parses_valid_yaml(tmp_path):
    """load_reddit_sources should return project_id and the reddit_sources list."""
    config_file = tmp_path / "sources.yaml"
    config_file.write_text(textwrap.dedent("""
        project_id: test-project
        sources: []
        reddit_sources:
          - name: r/test
            subreddit: test
            enabled: true
    """))

    config = load_reddit_sources(config_path=config_file)

    assert config["project_id"] == "test-project"
    assert len(config["reddit_sources"]) == 1
    assert config["reddit_sources"][0]["subreddit"] == "test"
    assert config["reddit_sources"][0]["enabled"] is True


def test_load_reddit_sources_raises_on_missing_file(tmp_path):
    """load_reddit_sources should raise FileNotFoundError for a non-existent path."""
    missing = tmp_path / "does_not_exist.yaml"

    with pytest.raises(FileNotFoundError, match="Sources config not found"):
        load_reddit_sources(config_path=missing)


# --------------------------------------------------------------------------- #
# fetch_reddit_posts tests                                                     #
# --------------------------------------------------------------------------- #

def test_fetch_reddit_posts_returns_posts_for_valid_response():
    """
    A mocked response with two external link posts should return two correctly
    structured dicts, each with headline, source_url, excerpt, and date_scraped.
    """
    posts = [
        {"title": "Headline One", "url": "https://example.com/article-1"},
        {"title": "Headline Two", "url": "https://example.com/article-2"},
    ]
    mock_response = _make_reddit_response(posts)

    with patch("reddit_scraper.requests.get", return_value=mock_response):
        results = fetch_reddit_posts(source={"name": "r/test", "subreddit": "test"})

    assert len(results) == 2

    assert results[0]["headline"] == "Headline One"
    assert results[0]["source_url"] == "https://example.com/article-1"
    assert results[0]["excerpt"] is None
    assert isinstance(results[0]["date_scraped"], datetime)

    assert results[1]["headline"] == "Headline Two"
    assert results[1]["source_url"] == "https://example.com/article-2"


def test_fetch_reddit_posts_skips_reddit_self_links():
    """
    Posts whose URL points back to reddit.com should be excluded — we only
    want external article URLs, not Reddit-internal links or self-posts.
    """
    posts = [
        {"title": "External Article", "url": "https://example.com/story"},
        {"title": "Reddit Self Post",  "url": "https://www.reddit.com/r/test/comments/abc/"},
    ]
    mock_response = _make_reddit_response(posts)

    with patch("reddit_scraper.requests.get", return_value=mock_response):
        results = fetch_reddit_posts(source={"name": "r/test", "subreddit": "test"})

    assert len(results) == 1
    assert results[0]["source_url"] == "https://example.com/story"


def test_fetch_reddit_posts_returns_empty_list_on_network_error():
    """A network-level exception should be caught and return an empty list."""
    import requests as req

    with patch("reddit_scraper.requests.get", side_effect=req.RequestException("timeout")):
        results = fetch_reddit_posts(source={"name": "r/test", "subreddit": "test"})

    assert results == []


# --------------------------------------------------------------------------- #
# scrape_reddit orchestration tests                                            #
# --------------------------------------------------------------------------- #

def test_scrape_reddit_adds_new_and_skips_duplicates(tmp_path):
    """
    scrape_reddit() should add posts whose URLs are not in the database,
    skip posts whose URLs are already present, and return an accurate summary.

    One source with two posts:
      - One URL already in the database (skipped)
      - One new URL (added)
    Expected: fetched=2, added=1, skipped=1, errors=0
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
            source_url="https://example.com/already",
            source_name="r/test",
            excerpt=None,
            date_scraped=datetime(2025, 1, 1),
        )

    # ------------------------------------------------------------------ #
    # Write a minimal sources.yaml for this test                           #
    # ------------------------------------------------------------------ #
    config_file = tmp_path / "sources.yaml"
    config_file.write_text(textwrap.dedent("""
        project_id: test-project
        sources: []
        reddit_sources:
          - name: r/test
            subreddit: test
            enabled: true
    """))

    # ------------------------------------------------------------------ #
    # Mock fetch_reddit_posts to return controlled entries                 #
    # ------------------------------------------------------------------ #
    fake_posts = [
        {"headline": "New Post",       "source_url": "https://example.com/new",     "excerpt": None, "date_scraped": datetime(2025, 6, 1)},
        {"headline": "Already Stored", "source_url": "https://example.com/already", "excerpt": None, "date_scraped": datetime(2025, 6, 1)},
    ]

    with patch("reddit_scraper.fetch_reddit_posts", return_value=fake_posts), \
         patch("reddit_scraper.get_engine", return_value=engine), \
         patch("reddit_scraper.init_db"):

        summary = scrape_reddit(config_path=config_file, db_path=":memory:")

    assert summary["fetched"] == 2
    assert summary["added"] == 1
    assert summary["skipped"] == 1
    assert summary["errors"] == 0

    # Verify the new article is actually in the database.
    with get_session(engine) as session:
        repo = ArticleRepository(session)
        candidates = repo.get_candidates(project_id="test-project")
        urls = {a.source_url for a in candidates}
        count = len(candidates)

    # Two total: the one pre-inserted + one new one.
    assert count == 2
    assert "https://example.com/new" in urls
