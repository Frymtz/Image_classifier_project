import logging
from datetime import datetime
import os

class Logger:
    """
    Utility class to simplify logging usage in the terminal.
    Allows configuring the log level and using different named loggers.
    """

    def __init__(self, name: str = None, level: int = logging.INFO):
        """
        Initializes the logger.
        :param name: Logger name (None for root logger).
        :param level: Minimum message level (e.g., logging.INFO).
        """
        self.logger = logging.getLogger(name)
        if not self.logger.hasHandlers():
            # Console handler
            stream_handler = logging.StreamHandler()
            stream_formatter = logging.Formatter('[%(levelname)s] %(message)s')
            stream_handler.setFormatter(stream_formatter)
            self.logger.addHandler(stream_handler)

            # File handler
            file_handler = logging.FileHandler('LOG_TXT/log.txt', encoding='utf-8')
            file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            file_handler.setFormatter(file_formatter)
            # Write header with date and time at the start of the log file
            log_file_path = 'LOG_TXT/log.txt'
            if not os.path.exists('LOG_TXT'):
                os.makedirs('LOG_TXT')
            # Overwrite file if it's the first logger instance in this run
            if not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0:
                with open(log_file_path, 'w', encoding='utf-8') as f:
                    f.write(f"===== Log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
            file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        self.logger.setLevel(level)
        self.logger.propagate = False  

    def info(self, msg, *args, **kwargs):
        """Logs an info message."""
        self.logger.info(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Logs an error message."""
        self.logger.error(msg, *args, **kwargs)

