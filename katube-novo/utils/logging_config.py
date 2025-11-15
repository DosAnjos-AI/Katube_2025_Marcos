"""
Configuracao de logging padronizado.
Formato: [YYYY-MM-DD HH:MM:SS] [LEVEL] [module.py:line] Mensagem
"""
import logging
from pathlib import Path
from typing import Optional
import sys


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configura logging padronizado do projeto.

    Formato padrao:
    [2025-11-14 10:30:45] [INFO] [pipeline.py:123] Mensagem aqui

    Args:
        level: Nivel de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path para arquivo de log (opcional)
        format_string: Formato customizado (opcional)

    Returns:
        Logger raiz configurado

    Exemplo:
        >>> logger = setup_logging(level="INFO")
        >>> logger.info("[INFO] Teste de logging")
        >>> logger.level <= logging.INFO
        True
    """
    if format_string is None:
        format_string = '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s'

    # Converter nivel de string para constante logging
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S'))

    handlers = [console_handler]

    # Handler para arquivo (se especificado)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Rotacao de logs: max 10MB por arquivo, 5 backups
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S'))
        handlers.append(file_handler)

    # Configurar root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True  # Substitui configuracao existente
    )

    root_logger = logging.getLogger()
    root_logger.info(f"[CONFIG] Logging configurado - nivel: {level}")
    if log_file:
        root_logger.info(f"[CONFIG] Log file: {log_file}")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Retorna logger configurado para modulo especifico.

    Args:
        name: Nome do modulo (__name__)

    Returns:
        Logger configurado para o modulo

    Exemplo:
        >>> logger = get_logger(__name__)
        >>> logger.name
        'utils.logging_config'
    """
    return logging.getLogger(name)


def configure_third_party_loggers(level: str = "WARNING"):
    """
    Configura nivel de log para bibliotecas de terceiros ruidosas.

    Reduz verbosidade de bibliotecas como urllib3, matplotlib, etc.

    Args:
        level: Nivel de log para terceiros (padrao: WARNING)

    Exemplo:
        >>> configure_third_party_loggers("ERROR")
        >>> logging.getLogger("urllib3").level >= logging.ERROR
        True
    """
    numeric_level = getattr(logging, level.upper(), logging.WARNING)

    noisy_loggers = [
        'urllib3',
        'matplotlib',
        'PIL',
        'torch',
        'transformers',
        'pyannote',
        'werkzeug',
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(numeric_level)


def disable_logging():
    """
    Desabilita completamente o logging.

    Util para testes ou quando nao se quer nenhuma saida.

    Exemplo:
        >>> disable_logging()
        >>> logging.getLogger().level
        50
    """
    logging.disable(logging.CRITICAL)


def enable_logging():
    """
    Reabilita o logging apos disable_logging().

    Exemplo:
        >>> disable_logging()
        >>> enable_logging()
        >>> logging.getLogger().level < 50
        True
    """
    logging.disable(logging.NOTSET)


def log_exception(logger: logging.Logger, exception: Exception, message: str = ""):
    """
    Loga excecao com traceback completo.

    Args:
        logger: Logger a usar
        exception: Excecao capturada
        message: Mensagem adicional (opcional)

    Exemplo:
        >>> logger = get_logger(__name__)
        >>> try:
        ...     raise ValueError("Teste")
        ... except ValueError as e:
        ...     log_exception(logger, e, "Erro de teste")
    """
    if message:
        logger.error(f"[ERRO] {message}")
    logger.error(f"[ERRO] {type(exception).__name__}: {str(exception)}")
    logger.exception(exception)
