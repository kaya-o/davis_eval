import argparse
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"


def validate_name(name):
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"suite name must be a simple directory name, got {name!r}")


def create_suite(suite_name, results_root=DEFAULT_RESULTS_ROOT):
    validate_name(suite_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = Path(results_root) / f"suite_{timestamp}_{suite_name}"
    suite_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "suite_name": suite_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with (suite_dir / "suite_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    return suite_dir


def main():
    parser = argparse.ArgumentParser(description="Create a timestamped experiment suite directory.")
    parser.add_argument("suite_name", help="Human-readable suite name, e.g. beta_sweep.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory for suites. Defaults to results/.",
    )
    args = parser.parse_args()

    suite_dir = create_suite(args.suite_name, results_root=args.results_root)
    print(suite_dir)


if __name__ == "__main__":
    main()
