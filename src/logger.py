"""
logger.py — the CCTV of the system.

Rubric item 3 (Logging and Monitoring): this is the logging half.

Why not print()?
  * print has no timestamp, no severity level, and cannot be filtered
  * once the app is inside a container, print is just loose text in
    `docker logs` that nothing can parse

What we produce instead: one JSON object per line, one line per event.
Because it is JSON, the log file is directly readable as a table:

    pandas.read_json("logs/predictions.log", lines=True)

That is exactly how src/monitor.py reads production traffic back.
"""

import json
import logging
import os

from src import config


class JsonFormatter(logging.Formatter):
    """Turn one log record into one line of JSON."""

    def format(self, record):
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "event": record.getMessage(),
        }

        # Anything passed in through extra={...} becomes a top-level field.
        # We work out which attributes are "extra" by comparing against a
        # blank record, rather than maintaining a hand-written list that
        # would drift out of date.
        blank = logging.LogRecord("", 0, "", 0, "", (), None)
        standard = set(blank.__dict__.keys())
        standard.add("message")
        standard.add("asctime")
        standard.add("taskName")

        for key in record.__dict__:
            if key not in standard:
                payload[key] = record.__dict__[key]

        return json.dumps(payload, ensure_ascii=False)


def get_logger(name="jakarta-housing"):
    """A logger that writes to the screen AND to logs/predictions.log."""
    log = logging.getLogger(name)

    # Guard against adding the same handlers twice. uvicorn --reload imports
    # this module more than once, and without the guard every line would
    # appear in duplicate.
    if len(log.handlers) > 0:
        return log

    log.setLevel(logging.INFO)

    folder = os.path.dirname(config.LOG_FILE)
    if folder != "":
        os.makedirs(folder, exist_ok=True)

    to_file = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    to_file.setFormatter(JsonFormatter())
    log.addHandler(to_file)

    to_screen = logging.StreamHandler()
    to_screen.setFormatter(JsonFormatter())
    log.addHandler(to_screen)

    log.propagate = False
    return log
