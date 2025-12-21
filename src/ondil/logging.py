from typing import Callable, Literal, Optional, Union, get_args
from uuid import UUID
import sys

from loguru._logger import Core as _Core, Logger as _Logger

LOG_LEVEL = Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]

LogFormat = (
    "<fg #B0BEC5>{time:YYYY-MM-DD HH:mm:ss.SSS}</fg #B0BEC5> | "
    "<level>{level: <8}</level> | "
    "<fg #E91E63>{process.name: <11}</fg #E91E63> | "
    "<fg #E91E63>{thread.name: <10}</fg #E91E63> | "
    "<fg #2196F3>{name}</fg #2196F3>:"
    "<fg #03A9F4>{function}</fg #03A9F4>:"
    "<fg #009688>{line}</fg #009688> - "
    "<level>{message}</level>"
)

# Create an independent Loguru logger instance for this package
logger = _Logger(
    core=_Core(),
    exception=None,
    depth=0,
    record=False,
    lazy=False,
    colors=False,
    raw=False,
    capture=True,
    patchers=[],
    extra={},
)

# No default sink - will be added by set_log_level() when set_config() is called
_handler_id: Optional[int] = None


def set_log_level(
    level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
) -> None:
    """
    Change the log level of the entsoe logger.

    Args:
        level: Log level (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)

    Raises:
        ValueError: If an invalid log level is provided
    """
    global _handler_id

    # Validate log level (runtime check, as Literal is only for type checking)
    valid_levels = tuple(
        get_args(
            Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
        )
    )
    if level not in valid_levels:
        raise ValueError(f"Invalid log_level '{level}'. Must be one of: {valid_levels}")

    # Remove the current handler if it exists
    if _handler_id is not None:
        try:
            logger.remove(_handler_id)
        except ValueError:
            # Handler doesn't exist, that's fine
            pass

    # Add a new handler with the updated level
    _handler_id = logger.add(
        sink=sys.stderr,
        level=level,
        colorize=True,
        format=LogFormat,
    )
    logger.success(f"Log level set to {level}")
