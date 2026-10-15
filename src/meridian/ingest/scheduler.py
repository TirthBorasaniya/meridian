"""Daily scheduler for incremental ingestion, run by the `scheduler` compose service.

APScheduler fires `meridian ingest --since last_success` once a day at the configured time; at
most one run is active, and a missed run is coalesced rather than stacked.
"""

import logging

from meridian.common.config import SchedulerConfig, get_scheduler_config

logger = logging.getLogger(__name__)


def daily_job(deadline_minutes: float) -> None:
    """Run one gated increment against the live collection and log its decision."""
    from meridian.ingest.pipeline import DeadlineExceededError, default_context, run_increment

    try:
        record = run_increment(
            default_context(), since="last_success", deadline_minutes=deadline_minutes
        )
    except DeadlineExceededError as exc:
        logger.error("scheduled ingest aborted", extra={"error": str(exc)})
        return
    except RuntimeError as exc:
        logger.error("scheduled ingest could not start", extra={"error": str(exc)})
        return
    logger.info(
        "scheduled ingest finished",
        extra={
            "batch_id": record.batch_id,
            "decision": record.decision,
            "papers_indexed": record.papers_indexed,
            "chunks_added": record.chunks_added,
        },
    )


def build_scheduler(scheduler_config: SchedulerConfig, o_blocking: bool = True):
    """
    Return an APScheduler scheduler with the daily ingestion job registered.

    Parameters
    ----------
    scheduler_config : SchedulerConfig
        Time of day, timezone, misfire grace, and run deadline.
    o_blocking : bool
        Blocking scheduler for the container; background scheduler otherwise.

    Returns
    -------
    scheduler : BaseScheduler
        Scheduler with one job, not started.
    """
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    hour_text, minute_text = scheduler_config.daily_time.split(":")
    scheduler_class = BlockingScheduler if o_blocking else BackgroundScheduler
    scheduler = scheduler_class(timezone=scheduler_config.timezone)
    scheduler.add_job(
        daily_job,
        trigger=CronTrigger(
            hour=int(hour_text), minute=int(minute_text), timezone=scheduler_config.timezone
        ),
        kwargs={"deadline_minutes": scheduler_config.deadline_minutes},
        id="daily_ingest",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=scheduler_config.misfire_grace_seconds,
        replace_existing=True,
    )
    return scheduler


def cli_scheduler(args) -> int:
    """`meridian scheduler`: block and run the daily job."""
    scheduler_config = get_scheduler_config()
    logger.info(
        "scheduler starting",
        extra={"daily_time": scheduler_config.daily_time, "timezone": scheduler_config.timezone},
    )
    build_scheduler(scheduler_config).start()
    return 0
