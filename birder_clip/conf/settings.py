import logging.config
import os
from typing import Any

# Logging
# https://docs.python.org/3/library/logging.config.html
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOGGING: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "[{asctime}.{msecs:04.0f} {levelname} {filename}:{lineno:<4d}] {message}",
            "style": "{",
            "datefmt": "%d/%b/%Y %H:%M:%S",
        },
        "simple": {"format": "[{asctime} {levelname}] {message}", "style": "{"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "level": "DEBUG", "formatter": "verbose"},
    },
    "loggers": {
        "birder_clip": {"handlers": ["console"], "level": LOG_LEVEL, "propagate": False},
    },
}

logging.config.dictConfig(LOGGING)
