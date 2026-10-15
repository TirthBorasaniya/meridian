"""Write the private manual-review sheets to `.planning/review/` (gitignored).

Usage::

    uv run python scripts/prepare_review.py
"""

from meridian.eval.review_prep import MissingSourceError, main

if __name__ == "__main__":
    try:
        main()
    except MissingSourceError as exc:
        raise SystemExit(f"cannot prepare the review sheets: {exc}") from exc
