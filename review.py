"""
CLI review tool for the 202x Headlines article pipeline.

Presents candidate articles one at a time and records accept/reject/skip
decisions. Rejection prompts for one or more structured categories plus an
optional free-text note — both are stored for future ML filter training.

Usage:
    python review.py --project 202x-headlines
"""

import argparse
import sys

import readchar

from core import get_engine, get_session, init_db
from core.repository import ArticleRepository


# --------------------------------------------------------------------------- #
# Rejection categories                                                         #
# --------------------------------------------------------------------------- #

# Each entry is (display_number, short_key, label).
# short_key is the value stored in rejection_reason (comma-joined for multiples).
REJECTION_CATEGORIES = [
    (1, "not_absurd_enough",      "Not absurd enough       — headline does not land as accidental satire"),
    (2, "wrong_kind_of_dark",     "Wrong kind of dark      — cruel or depressing without wit"),
    (3, "too_on_the_nose",        "Too on-the-nose         — already reads as deliberate satire, not accidental"),
    (4, "not_capitalism_relevant","Not capitalism-relevant — absurd but does not reveal anything about the current moment"),
    (5, "source_untrustworthy",   "Source untrustworthy    — outlet not reliable enough to publish"),
    (6, "duplicate_covered",      "Duplicate / covered     — same story or angle already accepted"),
]

# Maps display number → short_key for quick lookup during input parsing.
CATEGORY_BY_NUMBER = {num: key for num, key, _ in REJECTION_CATEGORIES}


# --------------------------------------------------------------------------- #
# Argument parsing                                                             #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    """
    Parse and validate command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with a .project attribute.
    """
    parser = argparse.ArgumentParser(
        description="Review candidate articles for a 202x Headlines project."
    )
    parser.add_argument(
        "--project",
        required=True,
        metavar="PROJECT_ID",
        help='Project namespace to review, e.g. "202x-headlines".',
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Display helpers                                                              #
# --------------------------------------------------------------------------- #

DIVIDER = "─" * 45


def print_header(project_id: str, total: int) -> None:
    """
    Print the session header showing project name and queue size.

    Parameters
    ----------
    project_id : The project namespace being reviewed.
    total      : Number of candidates in the queue at session start.
    """
    # Capitalise for display only — project_id stays lowercase internally.
    title = project_id.replace("-", " ").title()
    print(f"\n{title} — Review Queue")
    print(DIVIDER)
    print(f"{total} candidate{'s' if total != 1 else ''} in queue\n")


def print_article(article, position: int, total: int) -> None:
    """
    Print the formatted article block for a single review item.

    Parameters
    ----------
    article  : An Article ORM instance.
    position : 1-based position in the review queue (e.g. 3 of 12).
    total    : Total queue size at session start.
    """
    scraped = article.date_scraped.strftime("%Y-%m-%d")

    print(DIVIDER)
    print(f"[{position} of {total}]")
    print(f"Headline : {article.headline}")
    # source_name is nullable — fall back to a dash if absent.
    print(f"Source   : {article.source_name or '—'}")
    print(f"URL      : {article.source_url}")
    print(f"Scraped  : {scraped}")
    print(DIVIDER)
    print("[a] Accept   [r] Reject   [s] Skip   [q] Quit")


def print_rejection_menu() -> None:
    """Print the rejection category selection menu."""
    print("\n  Why are you rejecting this article?")
    print("  Select one or more numbers, separated by commas (e.g. 1,3):\n")
    for num, _, label in REJECTION_CATEGORIES:
        print(f"  {num}. {label}")
    print(
        "\n  Tip: choose every category that applies — multiple reasons give the"
        "\n  filter better training signal than a single one.\n"
    )


def print_summary(accepted: int, rejected: int, skipped: int) -> None:
    """
    Print the end-of-session summary.

    Parameters
    ----------
    accepted : Number of articles accepted this session.
    rejected : Number of articles rejected this session.
    skipped  : Number of articles skipped this session.
    """
    print(f"\n{DIVIDER}")
    print("Queue complete.")
    print(f"Accepted: {accepted}  Rejected: {rejected}  Skipped: {skipped}")
    print(DIVIDER)


# --------------------------------------------------------------------------- #
# Input helpers                                                                #
# --------------------------------------------------------------------------- #

def read_keypress() -> str:
    """
    Read a single keypress from stdin without requiring Enter.

    Returns
    -------
    str
        The character pressed, lowercased.
    """
    return readchar.readchar().lower()


def prompt_rejection_categories() -> list[str]:
    """
    Show the rejection category menu and collect a validated selection.

    Loops until the user enters at least one valid category number.
    Invalid input shows an error and re-displays the menu.

    Returns
    -------
    list[str]
        List of short_key strings for each selected category,
        e.g. ["not_absurd_enough", "wrong_kind_of_dark"].
    """
    valid_numbers = set(CATEGORY_BY_NUMBER.keys())

    while True:
        print_rejection_menu()
        raw = input("  Selection: ").strip()

        if not raw:
            print("\n  ✗ Please select at least one category.\n")
            continue

        # Parse comma-separated numbers, stripping whitespace around each.
        parts = [p.strip() for p in raw.split(",")]

        selected_keys = []
        error = None

        for part in parts:
            # Each part must be a digit in the valid range.
            if not part.isdigit() or int(part) not in valid_numbers:
                error = f"  ✗ '{part}' is not a valid option. Choose numbers from 1 to {len(valid_numbers)}.\n"
                break
            key = CATEGORY_BY_NUMBER[int(part)]
            # Ignore duplicates if the user enters e.g. "1,1".
            if key not in selected_keys:
                selected_keys.append(key)

        if error:
            print(f"\n{error}")
            continue

        return selected_keys


def prompt_optional_note() -> str | None:
    """
    Prompt for an optional free-text rejection note.

    Returns
    -------
    str | None
        The note text, or None if the user pressed Enter without typing.
    """
    note = input("  Optional note (press Enter to skip): ").strip()
    return note if note else None


# --------------------------------------------------------------------------- #
# Review loop                                                                  #
# --------------------------------------------------------------------------- #

def run_review_loop(repo: ArticleRepository, project_id: str) -> None:
    """
    Main review loop: present each candidate and process the user's decision.

    Runs until the candidate queue is exhausted or the user quits with [q].

    Parameters
    ----------
    repo       : An ArticleRepository bound to an active session.
    project_id : The project namespace being reviewed.
    """
    candidates = repo.get_candidates(project_id=project_id)
    total = len(candidates)

    print_header(project_id=project_id, total=total)

    if total == 0:
        print("No candidates to review.")
        return

    accepted = 0
    rejected = 0
    skipped = 0

    for position, article in enumerate(candidates, start=1):

        # ------------------------------------------------------------------ #
        # Display current article                                              #
        # ------------------------------------------------------------------ #

        print_article(article=article, position=position, total=total)

        # ------------------------------------------------------------------ #
        # Wait for a single keypress decision                                 #
        # ------------------------------------------------------------------ #

        while True:
            key = read_keypress()

            if key == "a":
                # ---------------------------------------------------------- #
                # Accept                                                       #
                # ---------------------------------------------------------- #
                repo.accept(article_id=article.id)
                print("\n✓ Accepted\n")
                accepted += 1
                break

            elif key == "s":
                # ---------------------------------------------------------- #
                # Skip — leave status unchanged, move on                      #
                # ---------------------------------------------------------- #
                print("\n→ Skipped\n")
                skipped += 1
                break

            elif key == "r":
                # ---------------------------------------------------------- #
                # Reject — collect categories then optional note              #
                # ---------------------------------------------------------- #
                selected_keys = prompt_rejection_categories()
                note = prompt_optional_note()

                # Store multiple categories as a comma-separated string so
                # the existing rejection_reason column needs no schema change.
                reason_string = ", ".join(selected_keys)

                # Human-readable category names for the confirmation line.
                key_to_label = {
                    key: label.split("—")[0].strip()
                    for _, key, label in REJECTION_CATEGORIES
                }
                display_names = ", ".join(
                    key_to_label[k] for k in selected_keys
                )

                repo.reject(
                    article_id=article.id,
                    reason=reason_string,
                    notes=note,
                )
                print(f"\n✗ Rejected — {display_names}\n")
                rejected += 1
                break

            elif key == "q":
                # ---------------------------------------------------------- #
                # Quit early — print summary and exit                         #
                # ---------------------------------------------------------- #
                print_summary(accepted=accepted, rejected=rejected, skipped=skipped)
                return

            # Any other key: silently ignore and wait for a valid input.

    # ---------------------------------------------------------------------- #
    # Queue exhausted                                                          #
    # ---------------------------------------------------------------------- #

    print_summary(accepted=accepted, rejected=rejected, skipped=skipped)


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main() -> None:
    """
    Entry point. Parses args, sets up the database, and starts the review loop.
    """
    args = parse_args()

    engine = get_engine()
    init_db(engine)

    with get_session(engine) as session:
        repo = ArticleRepository(session)
        run_review_loop(repo=repo, project_id=args.project)


if __name__ == "__main__":
    main()
