"""Structured JSON logging with the fields required by the conventions: ts, level, service, event.

Context ids (run_id, request_id, batch_id) are passed through `extra` and emitted as top-level
fields.
"""

import json
import logging
import sys
from datetime import UTC, datetime

# ============= Constants =============

_RESERVED_ATTR_SET = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "message",
    "asctime",
}


# ============= Formatter =============


class JsonFormatter(logging.Formatter):
    """Render each record as one JSON object per line."""

    def __init__(self, service: str) -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record with extras as top-level keys."""
        payload_dict = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname.lower(),
            "service": self._service,
            "event": record.getMessage(),
            "logger": record.name,
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTR_SET and not key.startswith("_"):
                payload_dict[key] = value
        if record.exc_info:
            payload_dict["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload_dict, default=str)


def configure_logging(service: str = "meridian", level: str = "INFO") -> None:
    """
    Install the JSON formatter on the root logger.

    Parameters
    ----------
    service : str
        Value of the `service` field on every record.
    level : str
        Root log level name.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter(service))
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level.upper())
    for noisy_name in ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy_name).setLevel(logging.WARNING)
