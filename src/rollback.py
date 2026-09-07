"""
rollback.py — move @champion back to the previous version.

Run:
    python -m src.rollback                    # back to the previous champion
    python -m src.rollback --to-version 3     # back to a specific version

Notice what is absent: no docker build, no deploy, no commit, no code change.
We move a label. That is the entire point of registry aliases.
"""

import argparse

import mlflow
from mlflow import MlflowClient

from src import config
from src.train import move_champion


def show_all_versions(client):
    """Print every registered version with its MAE and aliases."""
    # The source of truth for aliases is the registered model itself.
    # It comes back as a mapping like {"champion": "3", "challenger": "2"}.
    aliases_by_version = {}
    registered = client.get_registered_model(config.MODEL_NAME)
    for alias_name in registered.aliases:
        number = int(registered.aliases[alias_name])
        if number not in aliases_by_version:
            aliases_by_version[number] = []
        aliases_by_version[number].append("@" + alias_name)

    rows = []
    for version in client.search_model_versions("name='" + config.MODEL_NAME + "'"):
        run = client.get_run(version.run_id)
        mae = run.data.metrics.get("mae", 0.0)
        number = int(version.version)
        rows.append((number, run.info.run_name, mae,
                     aliases_by_version.get(number, [])))
    rows.sort()

    print("\n  version  run                       mae         aliases")
    print("  " + "-" * 70)
    for number, run_name, mae, aliases in rows:
        marker = ""
        if "@" + config.PRODUCTION_ALIAS in aliases:
            marker = "  <-- served by the API right now"
        print("  v{:<7d} {:<22s} $ {:>9,.0f}   {}{}".format(
            number, str(run_name)[:22], mae, " ".join(aliases), marker))

    return rows


def main():
    parser = argparse.ArgumentParser(description="Roll the champion back")
    parser.add_argument("--to-version", type=int, default=None)
    args = parser.parse_args()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()

    print("=" * 70)
    print("ROLLBACK — moving the @" + config.PRODUCTION_ALIAS + " alias")
    print("=" * 70)

    show_all_versions(client)

    champion = client.get_model_version_by_alias(
        config.MODEL_NAME, config.PRODUCTION_ALIAS)
    current_version = int(champion.version)

    target = args.to_version
    if target is None:
        # Default: go back to the previous CHAMPION, which is not
        # necessarily "current version minus one" — the version before this
        # one may never have been good enough to serve.
        try:
            previous = client.get_model_version_by_alias(
                config.MODEL_NAME, config.PREVIOUS_ALIAS)
            target = int(previous.version)
            print("\n  @" + config.PREVIOUS_ALIAS + " points at v" + str(target))
        except Exception:
            raise SystemExit(
                "No previous champion has been recorded yet.\n"
                "   Name the target explicitly, for example:\n"
                "   python -m src.rollback --to-version 1")

    if target == current_version:
        raise SystemExit("v" + str(target) + " is already the champion.")

    print("\n  ROLLBACK: v" + str(current_version) + "  ->  v" + str(target))
    move_champion(client, target)

    print("\n  The API still holds the old model in memory — it was loaded")
    print("  once at startup. Tell it to pick up the change:")
    print("      curl -X POST http://localhost:8000/admin/reload")


if __name__ == "__main__":
    main()
