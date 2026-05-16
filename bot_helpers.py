import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def silence_noisy_dependencies() -> None:
    """Silence optional noisy library warnings during startup."""
    # No-op by default; adjust if you have noisy third-party libraries.
    return


def check_environment(strict: bool = True) -> None:
    """Validate the environment before running the bot."""
    if not strict:
        return

    missing: list[str] = []
    # Add required environment variables if needed by your bot.
    required_env_vars = []
    for env_var in required_env_vars:
        if os.getenv(env_var) is None:
            missing.append(env_var)

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


def print_startup_banner(run_mode: str, will_publish: bool) -> None:
    """Display a basic startup banner."""
    logger.info(
        f"Starting Metaculus bot in mode={run_mode}, publish_reports={will_publish}"
    )
    print(f"Starting bot in {run_mode} mode. publish_reports={will_publish}")


def print_run_summary_banner(
    forecast_reports: Any,
    will_publish: bool,
    tournament_url: str | None = None,
) -> None:
    """Display a basic run summary banner."""
    logger.info(
        f"Run complete. publish_reports={will_publish}, tournament_url={tournament_url}"
    )
    print("Run complete.")
    if tournament_url:
        print(f"Tournament URL: {tournament_url}")
