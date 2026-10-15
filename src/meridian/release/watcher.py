"""Background release watcher for the demo: every interval, restore a newer accepted snapshot
when `latest.json` points at one, then refresh the serving runtime's index information."""

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class ReleaseWatcher:
    """Daemon thread that checks for newer releases on a fixed interval."""

    def __init__(self, runtime, interval_s: float, work_dir: Path, restore=None) -> None:
        from meridian.release.restore import restore_latest

        self.runtime = runtime
        self.interval_s = interval_s
        self.work_dir = work_dir
        self._restore = restore or restore_latest
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="release-watcher", daemon=True)

    def check_once(self) -> dict | None:
        """Restore the latest release if it is not live; return what was restored."""
        current = self.runtime.index_info(o_refresh=True).snapshot
        info = self._restore(self.work_dir, o_install_manifest=False, current_snapshot=current)
        if info is not None:
            self.runtime.index_info(o_refresh=True)
            logger.info("newer release restored", extra={"snapshot": info["snapshot"]})
        return info

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.check_once()
            except (RuntimeError, OSError, ValueError) as exc:
                logger.error("release check failed", extra={"error": str(exc)})

    def start(self) -> None:
        """Start the thread."""
        self._thread.start()

    def stop(self) -> None:
        """Ask the thread to stop."""
        self._stop.set()
