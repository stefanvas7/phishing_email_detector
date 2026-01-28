"""
In any module, do:

    from src.phishing_email_detector.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Starting training...")

To setup consistent logging throughout the whole project, in the main script (or in any testing scripts) or CLI entry point, call:

    from src.phishing_email_detector.utils.logging import configure_logging
    
    config = LoggingConfig(
        level="DEBUG",
        console_level="DEBUG",
        file_level="DEBUG",
        log_dir="logs",
        log_file_name="phishing_detector.log",
        max_bytes=10 * 1024 * 1024,  # 10 MB
        backup_count=5
    )

    configure_logging()  # once, at startup

This will create a root logger with the specified configuration.
"""

from __future__ import annotations

import logging
import logging.handlers
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass
class LoggingConfig:
    """
    Configuration for project-wide logging.

    Attributes
    ----------
    level:
        Global logging level for the root logger.
        Typical values: "DEBUG", "INFO", "WARNING", "ERROR".
        - Use "DEBUG" while developing locally.
        # - Use "INFO" or above in production / cloud.

    console_level:
        Minimum level for console logs (stderr/stdout).
        Example: root level DEBUG, console_level INFO → debug logs only in file.

    file_level:
        Minimum level for file logs.

    log_dir:
        Directory where log files are stored.
        The directory is created automatically if it does not exist.

    log_file_name:
        Base name of the log file. RotatingFileHandler will create
        "phishing_detector.log", "phishing_detector.log.1", etc.

    max_bytes:
        Maximum size of a single log file in bytes before rotation.
        When the size exceeds this value, the logs are rotated.

    backup_count:
        Maximum number of rotated backup files to keep.
        Older files are deleted after this number is reached.

    fmt:
        Log message format string. The default includes:
        - Timestamp
        - Logger name
        - Log level
        - Process ID
        - Thread name
        - Message text

    datefmt:
        Date/time format for the timestamp in each log entry.

    structured:
        Reserved flag for future extension (e.g., JSON logging).
        For now, only plain text formatting is used.
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    console_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    file_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"

    log_dir: str = "logs"
    log_file_name: str = "phishing_detector.log"
    max_bytes: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5

    fmt: str = (
        "%(asctime)s | %(levelname)-8s | %(name)s | "
        "pid=%(process)d | thread=%(threadName)s | %(message)s"
    )
    datefmt: str = "%Y-%m-%d %H:%M:%S"

    structured: bool = False  # for future JSON/structured logging


# ------------------------------
# Internal state
# ------------------------------

# Module-level flag to prevent configuring logging multiple times.
# Python's logging module is global; calling configure_logging() repeatedly
# can cause duplicated handlers and repeated log lines.
_CONFIGURED = False


# ------------------------------
# Public API
# ------------------------------

def configure_logging(config: Optional[LoggingConfig] = None) -> None:
    """
    Configure the global logging system for the entire project.

    This function should be called ONCE, as early as possible
    in your main script or CLI entry point.

    Parameters
    ----------
    config:
        Optional LoggingConfig object. If None, a default configuration
        is used (suitable for local development).

    Behavior
    --------
    - Creates a root logger with the given log level.
    - Attaches two handlers:
        1) StreamHandler (console output)
        2) RotatingFileHandler (file with rotation)
    - Sets a consistent formatter for all handlers.
    - Ensures no duplicate handlers are added if called multiple times.

    Why this design?
    ----------------
    - Centralized configuration avoids misconfigured logging in different files.
    - Using the root logger allows all libraries and modules to inherit behavior.
    - Separate console and file handlers give you fine-grained control
      without complicating logging usage in modules.
    """
    global _CONFIGURED

    if _CONFIGURED:
        # Prevent duplicate handlers when configure_logging is called multiple times.
        # This can happen if a training script calls it and a notebook re-imports it.
        return

    if config is None:
        config = LoggingConfig()

    # 1. Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(config.level)

    # 2. Set up formatter (defines how each log line looks)
    formatter = logging.Formatter(fmt=config.fmt, datefmt=config.datefmt)

    # 3. Console handler (prints to stderr/stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(config.console_level)
    console_handler.setFormatter(formatter)

    # 4. File handler with rotation
    log_dir_path = Path(config.log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    log_file_path = log_dir_path / config.log_file_name
    
    # if max_bytes is exceded file is closed and new file is opened for output
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file_path),
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(config.file_level)
    file_handler.setFormatter(formatter)

    # 5. Attach handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # 6. Optional: reduce verbosity of noisy third-party loggers
    #    (TensorFlow, urllib3, etc.)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for a given module or component.

    Parameters
    ----------
    name:
        Usually use __name__ from the calling module, e.g.
            logger = get_logger(__name__)
        If None, the root logger is returned.

    Returns
    -------
    logging.Logger
        A configured logger instance.

    Example
    -------
        from phishing_email_detector.utils.logging import get_logger

        logger = get_logger(__name__)

        def train():
            logger.info("Starting training loop...")
            logger.debug("Batch 1/100, loss=0.1234")
    """
    return logging.getLogger(name)


def configure_for_notebook(level: str = "INFO") -> None:
    """
    Convenience function to configure logging for Jupyter/Colab notebooks.

    Differences from `configure_logging()`:
    - Only sets up a console handler (no file logging).
    - Slightly simpler format for easier reading in notebooks.
    - Can be called multiple times without duplicating handlers.

    Parameters
    ----------
    level:
        Logging level for the notebook environment.

    """
    global _CONFIGURED

    # In notebooks, we allow reconfiguration (e.g., changing level mid-session).
    # So we reset previous handlers but keep the original API.
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    # Keep TensorFlow quieter in notebooks, too.
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _CONFIGURED = True
