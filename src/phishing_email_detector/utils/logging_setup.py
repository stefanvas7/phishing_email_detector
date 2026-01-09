from src.phishing_email_detector.utils.logging import configure_logging, LoggingConfig, get_logger

"""
Setup logging for a run with default configuration.
Copy import and code below into every script run
"""
config = LoggingConfig(
    level="DEBUG",
    console_level="DEBUG",
    file_level="DEBUG",
    log_dir="logs",
    log_file_name="phishing_detector.log",
    max_bytes=10 * 1024 * 1024,  # 10 MB
    backup_count=5
)
configure_logging(config)