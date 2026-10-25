import logging.config
from typing import Any

from birder.conf import settings as birder_settings

# Paths
TOKENIZERS_DIR = birder_settings.MODELS_DIR.joinpath("tokenizers")

# Model registry
REGISTRY_BASE_URL = "https://f000.backblazeb2.com/file/birder-clip/models"

# Logging
# https://docs.python.org/3/library/logging.config.html
_LOGGING: dict[str, Any] = {
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
        "birder_clip": {"handlers": ["console"], "level": birder_settings.LOG_LEVEL, "propagate": False},
    },
}

logging.config.dictConfig(_LOGGING)
