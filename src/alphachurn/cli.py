"""Generate data, train the comparison, or score a snapshot."""

import argparse
import sys

from alphachurn.config import DEFAULT_SCORE_AS_OF, Settings, load_settings
from alphachurn.generate import generate_dataset
from alphachurn.score import score_as_of
from alphachurn.train import run_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="alphachurn",
        description="Customer churn prediction and explainability",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Write a local SQLite subscription history")
    generate.add_argument("--customers", type=int)
    generate.add_argument("--seed", type=int)

    train = sub.add_parser("train", help="Walk-forward train logistic regression and XGBoost")
    train.add_argument("--min-train-periods", type=int)

    score = sub.add_parser("score", help="Score active customers and write SHAP reasons")
    score.add_argument("--as-of", default=DEFAULT_SCORE_AS_OF)
    score.add_argument("--threshold", type=float)

    args = parser.parse_args(argv)
    settings = _settings_from_args(args)
    try:
        if args.command == "generate":
            counts = generate_dataset(settings)
            print(
                f"Wrote {counts['customers']} customers, "
                f"{counts['transactions']} transactions, "
                f"and {counts['events']} events to {settings.db_path}."
            )
            return 0
        if args.command == "train":
            result = run_training(settings)
            _print_training(result)
            return 0
        scored = score_as_of(settings, args.as_of)
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Scored {len(scored)} active customers as of {args.as_of}.")
    preview = scored.head(5)[
        ["customer_id", "plan_tier", "churn_probability", "reason_1_feature", "reason_1_shap"]
    ]
    print(preview.to_string(index=False))
    print(f"Wrote {settings.artifact_dir / 'scores.csv'}.")
    return 0


def _settings_from_args(args) -> Settings:
    settings = load_settings()
    updates = {}
    if getattr(args, "customers", None) is not None:
        updates["n_customers"] = args.customers
    if getattr(args, "seed", None) is not None:
        updates["seed"] = args.seed
    if getattr(args, "min_train_periods", None) is not None:
        updates["min_train_periods"] = args.min_train_periods
    if getattr(args, "threshold", None) is not None:
        updates["threshold"] = args.threshold
    if not updates:
        return settings
    return Settings(**{**settings.__dict__, **updates})


def _print_training(result) -> None:
    print(
        f"Labeled rows: {result.n_rows}. "
        f"Share who churn in the next 30 days: {result.positive_rate:.3f}."
    )
    print(f"Tuning snapshot: {result.tune_valid_as_of.date().isoformat()}.")
    print(f"XGBoost parameters: {result.params}")
    print(result.metrics.to_string(index=False))
    print()
    print(result.summary.round(3).to_string())
    print(f"Saved models, fold metrics, and SHAP importance under {result.artifact_dir}.")


if __name__ == "__main__":
    raise SystemExit(main())
