import logging

class Logger:
    """
    Classe utilitária para facilitar o uso do logging no terminal.
    Permite configurar o nível de log e usar diferentes loggers nomeados.
    """

    def __init__(self, name: str = None, level: int = logging.INFO):
        """
        Inicializa o logger.
        :param name: Nome do logger (None para root logger).
        :param level: Nível mínimo das mensagens (ex: logging.INFO).
        """
        self.logger = logging.getLogger(name)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(level)
        self.logger.propagate = False  # Evita mensagens duplicadas

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
