"""Entrypoint for the RAGAS evaluation suite.

Runs the compiled graph over the evaluation question set and scores faithfulness
and context precision against the chunks the retriever actually returned. Use
``--num-questions`` to run a smaller smoke subset (20 is recommended before the
full 75-question suite).

Examples
--------
Run a 20-question smoke evaluation::

    python scripts/evaluate.py --num-questions 20

Run the full 75-question suite::

    python scripts/evaluate.py
"""

import argparse

from meridian.evaluation.ragas_runner import run_evaluation


def main() -> None:
    """Parse arguments and run the evaluation suite."""
    parser = argparse.ArgumentParser(description="Run the Meridian RAGAS evaluation suite.")
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Number of questions to evaluate. Defaults to the full 75-question set.",
    )
    args = parser.parse_args()

    score_dict = run_evaluation(num_questions=args.num_questions)
    print("RAGAS scores:")
    for metric_name, value in score_dict.items():
        print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
