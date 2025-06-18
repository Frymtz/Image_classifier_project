import logging

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
            self.logger.addHandler(file_handler)

        self.logger.setLevel(level)
        self.logger.propagate = False  

    def info(self, msg, *args, **kwargs):
        """Loga uma mensagem de informação."""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Loga uma mensagem de aviso."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Loga uma mensagem de erro."""
        self.logger.error(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        """Loga uma mensagem de depuração."""
        self.logger.debug(msg, *args, **kwargs)
