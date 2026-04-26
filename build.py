"""
Static site generator for the 202x Headlines project.

Reads all accepted articles from the database, groups them by publication
year, and writes a folder of static HTML files to dist/. The output is
ready to upload to any web host with no server-side processing required.

Output structure:
    dist/
    ├── index.html      — landing page listing all available years
    ├── style.css       — copied from templates/style.css
    └── <year>.html     — one file per year, listing that year's headlines

Templates live in templates/ as plain HTML files. Variable substitution
uses str.replace() with {{PLACEHOLDER}} tokens — no templating framework.

Usage:
    python build.py
"""

import html
import shutil
from collections import defaultdict
from pathlib import Path

from core import get_engine, get_session, init_db
from core.repository import ArticleRepository


# --------------------------------------------------------------------------- #
# Paths and constants                                                          #
# --------------------------------------------------------------------------- #

_PROJECT_ROOT  = Path(__file__).resolve().parent
_TEMPLATES_DIR = _PROJECT_ROOT / "templates"
_DIST_DIR      = _PROJECT_ROOT / "dist"
_DB_PATH       = _PROJECT_ROOT / "data" / "articles.db"

# Project ID used to scope all repository queries.
_PROJECT_ID = "202x-headlines"


# --------------------------------------------------------------------------- #
# Template loading                                                             #
# --------------------------------------------------------------------------- #

def load_template(name: str) -> str:
    """
    Read a template file from the templates/ directory and return its contents.

    Parameters
    ----------
    name : Filename of the template, e.g. "index.html" or "year.html".

    Returns
    -------
    str
        Full text of the template file.

    Raises
    ------
    FileNotFoundError
        If the template file does not exist.
    """
    path = _TEMPLATES_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Template not found: {path}\n"
            "Expected template files in the templates/ directory."
        )
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# HTML rendering helpers                                                       #
# --------------------------------------------------------------------------- #

def render_year_list_item(year: int, count: int) -> str:
    """
    Render one <li> entry for the index page year list.

    Parameters
    ----------
    year  : Four-digit publication year.
    count : Number of accepted headlines in that year.

    Returns
    -------
    str
        An indented <li> element linking to the year page.
    """
    # Noun agreement for "headline" vs "headlines"
    noun = "headline" if count == 1 else "headlines"
    return f'                <li><a href="{year}.html">{year} — {count} {noun}</a></li>'


def render_headline_item(headline: str, source_url: str, image_url: str | None = None) -> str:
    """
    Render one <li> entry for a year page headline list.

    When image_url is provided, a bare <img> tag is prepended to the link
    inside the list item. Styling is left to style.css — no inline styles
    are applied here.

    Parameters
    ----------
    headline   : The article headline text. HTML-escaped before insertion.
    source_url : The original article URL.
    image_url  : Optional header image URL. Omitted entirely when None.

    Returns
    -------
    str
        An indented <li> element containing an optional image and a link.
    """
    # Escape headline text to prevent any stray < > & characters from
    # breaking the HTML — headlines scraped from the web can contain anything.
    safe_headline = html.escape(headline)

    # Build the optional image tag. alt="" marks it as decorative so screen
    # readers skip it — the headline link is the meaningful content.
    img_tag = f'<img src="{html.escape(image_url)}" alt="">' if image_url else ""

    # target="_blank" opens the article in a new tab so the archive stays open.
    # rel="noopener noreferrer" is a security standard for external new-tab links.
    return (
        f'            <li>'
        f'{img_tag}'
        f'<a href="{source_url}" target="_blank" rel="noopener noreferrer">'
        f'{safe_headline}'
        f'</a></li>'
    )


# --------------------------------------------------------------------------- #
# Build                                                                        #
# --------------------------------------------------------------------------- #

def build(db_path: Path = _DB_PATH) -> dict:
    """
    Run the full static site build.

    Steps:
      1. Read all accepted articles from the database.
      2. Group by publication year (newest year first).
      3. Write dist/<year>.html for each year.
      4. Write dist/index.html listing all years.
      5. Copy templates/style.css to dist/style.css.

    Prints a one-line summary on completion.

    Parameters
    ----------
    db_path : Path to the SQLite database. Defaults to data/articles.db.

    Returns
    -------
    dict
        Build summary with keys: "years_built", "total_headlines".
    """
    # ------------------------------------------------------------------ #
    # Set up database connection                                           #
    # ------------------------------------------------------------------ #

    engine = get_engine(db_path=db_path)
    init_db(engine)

    # ------------------------------------------------------------------ #
    # Fetch all accepted articles and group by year                        #
    # ------------------------------------------------------------------ #

    with get_session(engine) as session:
        repo = ArticleRepository(session)
        articles = repo.get_accepted(project_id=_PROJECT_ID)

        # Group articles by publication_year. Articles without a year are
        # skipped — they should not exist for accepted articles, but we
        # guard defensively rather than crashing the build.
        by_year: dict[int, list] = defaultdict(list)
        for article in articles:
            if article.publication_year is not None:
                by_year[article.publication_year].append(
                    (article.headline, article.source_url, article.image_url)
                )

    # Sort years descending (newest first) for both index and navigation.
    sorted_years = sorted(by_year.keys(), reverse=True)
    total_headlines = sum(len(v) for v in by_year.values())

    # ------------------------------------------------------------------ #
    # Prepare the output directory                                         #
    # ------------------------------------------------------------------ #

    # Wipe and recreate dist/ on every build so stale year files don't
    # accumulate when articles are removed or years change.
    if _DIST_DIR.exists():
        shutil.rmtree(_DIST_DIR)
    _DIST_DIR.mkdir()

    # ------------------------------------------------------------------ #
    # Load templates                                                       #
    # ------------------------------------------------------------------ #

    index_tmpl = load_template("index.html")
    year_tmpl  = load_template("year.html")

    # ------------------------------------------------------------------ #
    # Write one <year>.html per year                                       #
    # ------------------------------------------------------------------ #

    for year in sorted_years:
        headlines = by_year[year]
        # Headlines are already ordered newest-first by get_accepted().
        headline_items = "\n".join(
            render_headline_item(headline, url, image_url)
            for headline, url, image_url in headlines
        )
        count = len(headlines)
        plural = "" if count == 1 else "s"

        page = (year_tmpl
            .replace("{{YEAR}}",             str(year))
            .replace("{{HEADLINE_LIST}}",    headline_items)
            .replace("{{HEADLINE_COUNT}}",   str(count))
            .replace("{{HEADLINE_PLURAL}}", plural)
        )

        out_path = _DIST_DIR / f"{year}.html"
        out_path.write_text(page, encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Write index.html                                                     #
    # ------------------------------------------------------------------ #

    year_items = "\n".join(
        render_year_list_item(year, len(by_year[year])) for year in sorted_years
    )
    year_count = len(sorted_years)
    year_plural = "" if year_count == 1 else "s"

    index_page = (index_tmpl
        .replace("{{YEAR_LIST}}",    year_items)
        .replace("{{TOTAL_COUNT}}", str(total_headlines))
        .replace("{{YEAR_COUNT}}",  str(year_count))
        .replace("{{YEAR_PLURAL}}", year_plural)
    )

    (_DIST_DIR / "index.html").write_text(index_page, encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Copy style.css into dist/                                            #
    # ------------------------------------------------------------------ #

    shutil.copy(_TEMPLATES_DIR / "style.css", _DIST_DIR / "style.css")

    # ------------------------------------------------------------------ #
    # Print summary                                                        #
    # ------------------------------------------------------------------ #

    print(
        f"Build complete — "
        f"{year_count} year{year_plural} built, "
        f"{total_headlines} headline{'' if total_headlines == 1 else 's'} total."
    )

    return {
        "years_built":      year_count,
        "total_headlines":  total_headlines,
    }


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    build()
