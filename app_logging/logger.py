import sys
from typing import Any, Optional

from loguru import logger as _loguru_logger

from .context import get_context_prefix


class ContextAwareLogger:
    def __init__(self):
        _loguru_logger.remove()
        _loguru_logger.add(sys.stderr, format=self._get_format_string(), level="DEBUG", enqueue=True)

    def _get_format_string(self) -> str:
        return "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - {message}"

    def _get_context_prefix(self) -> str:
        context_prefix = get_context_prefix()
        return f"{context_prefix}: " if context_prefix else ""

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        _loguru_logger.opt(depth=1).debug(f"{self._get_context_prefix()}{message}", *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        _loguru_logger.opt(depth=1).info(f"{self._get_context_prefix()}{message}", *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        _loguru_logger.opt(depth=1).warning(f"{self._get_context_prefix()}{message}", *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        _loguru_logger.opt(depth=1).error(f"{self._get_context_prefix()}{message}", *args, **kwargs)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        _loguru_logger.opt(depth=1).critical(f"{self._get_context_prefix()}{message}", *args, **kwargs)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        _loguru_logger.opt(depth=1).exception(f"{self._get_context_prefix()}{message}", *args, **kwargs)

    def trace(self, message: str, *args: Any, **kwargs: Any) -> None:
        _loguru_logger.opt(depth=1).trace(f"{self._get_context_prefix()}{message}", *args, **kwargs)

    def success(self, message: str, *args: Any, **kwargs: Any) -> None:
        _loguru_logger.opt(depth=1).success(f"{self._get_context_prefix()}{message}", *args, **kwargs)

    def remove(self) -> None:
        _loguru_logger.remove()

    def add(self, *args: Any, **kwargs: Any) -> None:
        _loguru_logger.add(*args, **kwargs)


logger = ContextAwareLogger()


def get_logger(name: Optional[str] = None) -> ContextAwareLogger:
    return logger

